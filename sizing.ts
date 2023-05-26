interface Device {
  bytes: number;
  bytesPerSecond: number;
  flopsDirty: number; // FMAs, not really FLOPs
}

interface ModelArchitecture {
  nContext: number;
  nEmbedding: number;
  nHeads: number;
  nLayers: number;
  roughParams: number;
}

class Model {
  linearLayerBytes: number;
  linearWholeBytes: number;
  linearFlopsDirty: number; // FMAs, not really FLOPs

  kvCacheLayerBytes: number;
  kvCacheWholeBytes: number;

  constructor(
    public architecture: ModelArchitecture,
    public bytesPerParam: number
  ) {
    const { nContext, nEmbedding, nLayers } = architecture;

    this.linearLayerBytes = (3 * 3 + 4) * nEmbedding ** 2 * bytesPerParam;
    this.linearWholeBytes = this.linearLayerBytes * nLayers;
    this.linearFlopsDirty = this.linearWholeBytes / bytesPerParam;

    this.kvCacheLayerBytes = 2 * nContext * nEmbedding * bytesPerParam;
    this.kvCacheWholeBytes = this.kvCacheLayerBytes * nLayers;

    const expectedLinearWholeBytes = architecture.roughParams * bytesPerParam;
    if (
      Math.abs(expectedLinearWholeBytes - this.linearWholeBytes) /
        expectedLinearWholeBytes >
      0.1
    ) {
      throw new Error("Calculated number of parameters is too far off");
    }
  }
}

function calculateScenarioSingleDeviceAlternating({
  model,
  device,
}: {
  model: Model;
  device: Device;
}) {
  // Store model weights and KV stores in device (eg. CPU) memory. For each KV store, perform one layer of attention. Take the outputs and run the linear parts of the model batched. Repeat for all layers.

  const nKvCaches = Math.floor(
    (device.bytes - model.linearWholeBytes) / model.kvCacheWholeBytes
  );
  if (nKvCaches < 1) {
    throw new Error(
      "Not enough memory to store model and at least one KV store"
    );
  }

  const linearMemoryBoundSeconds =
    model.linearWholeBytes / device.bytesPerSecond;
  const linearComputeBoundSeconds =
    (nKvCaches * model.linearFlopsDirty) / device.flopsDirty;
  const linearSeconds = Math.max(
    linearMemoryBoundSeconds,
    linearComputeBoundSeconds
  );
  const linearBoundReason =
    linearMemoryBoundSeconds > linearComputeBoundSeconds ? "memory" : "compute";

  const kvCacheSeconds =
    (nKvCaches * model.kvCacheWholeBytes) / device.bytesPerSecond;

  const totalSeconds = linearSeconds + kvCacheSeconds;
  const tokensPerSecondSequential = 1 / totalSeconds;
  const tokensPerSecondParallel = nKvCaches / totalSeconds;

  return {
    nKvCaches,
    linearMemoryBoundSeconds,
    linearComputeBoundSeconds,
    linearSeconds,
    linearBoundReason,
    kvCacheSeconds,
    totalSeconds,
    tokensPerSecondSequential,
    tokensPerSecondParallel,
  };
}

function calculateScenarioSplitLinearAndKv({
  model,
  kvDevice,
  linearDevice,
}: {
  model: Model;
  kvDevice: Device;
  linearDevice: Device;
}) {
  // Store model weights on one device (e.g. CPU) and KV stores on another (e.g. GPU). For each KV store, perform one layer of attention. Send the outputs to the other device and run the linear parts of the model batched. Repeat for all layers. At any time, one of the devices is idle.

  const nKvCaches = Math.floor(kvDevice.bytes / model.kvCacheWholeBytes);
  if (nKvCaches < 1) {
    throw new Error("Not enough memory for at least one KV store");
  }

  const linearMemoryBoundSeconds =
    model.linearWholeBytes / linearDevice.bytesPerSecond;
  const linearComputeBoundSeconds =
    (nKvCaches * model.linearFlopsDirty) / linearDevice.flopsDirty;
  const linearSeconds = Math.max(
    linearMemoryBoundSeconds,
    linearComputeBoundSeconds
  );
  const linearBoundReason =
    linearMemoryBoundSeconds > linearComputeBoundSeconds ? "memory" : "compute";

  const kvCacheSeconds =
    (nKvCaches * model.kvCacheWholeBytes) / kvDevice.bytesPerSecond;

  const totalSeconds = linearSeconds + kvCacheSeconds;
  const tokensPerSecondSequential = 1 / totalSeconds;
  const tokensPerSecondParallel = nKvCaches / totalSeconds;

  return {
    nKvCaches,
    linearMemoryBoundSeconds,
    linearComputeBoundSeconds,
    linearSeconds,
    linearBoundReason,
    kvCacheSeconds,
    totalSeconds,
    tokensPerSecondSequential,
    tokensPerSecondParallel,
  };
}

const llama7b: ModelArchitecture = {
  nContext: 512,
  nEmbedding: 4096,
  nHeads: 32,
  nLayers: 32,
  roughParams: 7e9,
};

const devices = {
  cpu: {
    bytes: 32e9,
    bytesPerSecond: 21e9,
    flopsDirty: 16 * 4e9, // FP32 FMA, not really FLOPs
  },
  gpu: {
    bytes: 8e9,
    bytesPerSecond: 450e9,
    flopsDirty: 81e12 / 2, // "Tensor Compute" from Wikipedia, assuming 2 per FMA
  },
};

const configurations: { name: string; calculate: (model: Model) => unknown }[] =
  [
    {
      name: "CPU only alternating",
      calculate: (model) =>
        calculateScenarioSingleDeviceAlternating({
          model,
          device: devices.cpu,
        }),
    },
    {
      name: "GPU only alternating",
      calculate: (model) =>
        calculateScenarioSingleDeviceAlternating({
          model,
          device: devices.gpu,
        }),
    },
    {
      name: "CPU linear, GPU KV alternating",
      calculate: (model) =>
        calculateScenarioSplitLinearAndKv({
          model,
          kvDevice: devices.gpu,
          linearDevice: devices.cpu,
        }),
    },
    {
      name: "GPU linear, CPU KV alternating",
      calculate: (model) =>
        calculateScenarioSplitLinearAndKv({
          model,
          kvDevice: devices.cpu,
          linearDevice: devices.gpu,
        }),
    },
    {
      name: "Dual GPU alternating",
      calculate: (model) =>
        calculateScenarioSplitLinearAndKv({
          model,
          kvDevice: devices.gpu,
          linearDevice: devices.gpu,
        }),
    },
  ];

for (const bytesPerParam of [1, 2, 4]) {
  console.log(`bytesPerParam=${bytesPerParam}`);

  const model = new Model(llama7b, bytesPerParam);

  for (const configuration of configurations) {
    try {
      console.log(configuration.name, configuration.calculate(model));
    } catch (e) {
      console.log(configuration.name, "ERROR", (e as any).message);
    }
  }
}
