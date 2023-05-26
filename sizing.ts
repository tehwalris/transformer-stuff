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
  // Store model weights and KV stores in device (eg. CPU) memory. For each KV
  // store, perform one layer of attention. Take the outputs and run the linear
  // parts of the model batched. Repeat for all layers.

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

function calculateScenarioSplitLinearAndKvAlternating({
  model,
  kvDevice,
  linearDevice,
}: {
  model: Model;
  kvDevice: Device;
  linearDevice: Device;
}) {
  // Store model weights on one device (e.g. CPU) and KV stores on another (e.g.
  // GPU). For each KV store, perform one layer of attention. Send the outputs
  // to the other device and run the linear parts of the model batched. Repeat
  // for all layers. At any time, one of the devices is idle.

  const nKvCaches = Math.floor(kvDevice.bytes / model.kvCacheWholeBytes);
  if (nKvCaches < 1) {
    throw new Error("Not enough memory for at least one KV store");
  }
  if (model.linearWholeBytes > linearDevice.bytes) {
    throw new Error("Not enough memory for model weights");
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

function calculateScenarioSplitLinearAndKvInterleaved({
  model,
  kvDevice,
  kvDeviceCount,
  linearDevice,
  linearDeviceCount,
}: {
  model: Model;
  kvDevice: Device;
  kvDeviceCount: number;
  linearDevice: Device;
  linearDeviceCount: number;
}) {
  // Store model weights on one device (e.g. CPU) and KV stores on another (e.g.
  // GPU). For half of the KV stores, perform one layer of attention. Send the
  // outputs to the other device and run the linear parts of the model batched.
  // In the meantime perform attention for the remaining KV stores. Repeat for
  // all layers. If the linear layers take exactly as long as attention, then
  // both devices will always be busy.
  //
  // When there are multiple linear devices, each linear device is assigned
  // some linear layers of the model. Each linear layer is assigned to exactly
  // one device. This strategy will not duplicate layers to increase throughput.
  // The first device gets the first few layers, the second device gets the next
  // few, and so on. The last device might have fewer layers than the others. We
  // "add noop layers" to that device so that all devices have an equal number of
  // layers. When the device is "processing" a noop layer, it will idle.
  //
  // When there are multiple KV devices, each device is assigned some subset of
  // the KV stores. Each KV store is assigned to exactly one device. KV stores
  // are not split or duplicated across devices.
  //
  // Consider the KV stores in a single "half" (A/B) of a single KV device,
  // containing KV stores numbered i from 0 to some n-1. Assume KV store 0 has
  // just processed layer j. Assume each linear device has k layers. KV store i
  // will send its output to linear device (i + j // k) % linearDeviceCount.
  // Initially each KV store i starts processing layer (i % linearDeviceCount) *
  // k. Each such layer is the first layer on its assigned linear device.

  const nKvCachesPerDevice = Math.floor(
    kvDevice.bytes / model.kvCacheWholeBytes
  );
  if (nKvCachesPerDevice < 2) {
    throw new Error(
      "Not enough memory for at least two KV stores per KV device"
    );
  }
  const nKvCachesTotal = nKvCachesPerDevice * kvDeviceCount;
  const nKvCachesA = Math.floor(nKvCachesPerDevice / 2);
  const nKvCachesB = nKvCachesPerDevice - nKvCachesA;
  if (nKvCachesA < 1 || nKvCachesB < 1) {
    throw new Error("Splitting KV stores failed");
  }

  if (Math.min(nKvCachesA, nKvCachesB) < linearDeviceCount) {
    throw new Error(
      "At least one of the KV cache groups is smaller than the number of linear devices, which would lead to idle linear devices"
    );
  }

  const maxPossibleLinearLayersPerDevice = Math.floor(
    linearDevice.bytes / model.linearLayerBytes
  );
  if (maxPossibleLinearLayersPerDevice < 1) {
    throw new Error(
      "Not enough memory for at least one linear layer per linear device"
    );
  }
  const nLinearLayersByDevice: number[] = Array(linearDeviceCount).fill(0);
  let assignedLinearLayers = 0;
  outerLinearLayerAssignmentLoop: while (true) {
    for (let i = 0; i < linearDeviceCount; i++) {
      if (assignedLinearLayers === model.architecture.nLayers) {
        break outerLinearLayerAssignmentLoop;
      }
      if (nLinearLayersByDevice[i] < maxPossibleLinearLayersPerDevice) {
        nLinearLayersByDevice[i]++;
        assignedLinearLayers++;
      } else {
        throw new Error(
          "Not enough memory across linear devices to fit all linear layers"
        );
      }
    }
  }
  if (Math.min(...nLinearLayersByDevice) < 1) {
    throw new Error("Some linear devices have no linear layers");
  }

  const linearMemoryBoundSeconds =
    model.linearWholeBytes / linearDevice.bytesPerSecond;
  const nKvCachesPerLinearDeviceA =
    Math.ceil(nKvCachesA / linearDeviceCount) * kvDeviceCount;
  const nKvCachesPerLinearDeviceB =
    Math.ceil(nKvCachesB / linearDeviceCount) * kvDeviceCount;
  const linearComputeBoundSecondsA =
    (nKvCachesPerLinearDeviceA * model.linearFlopsDirty) /
    linearDevice.flopsDirty;
  const linearComputeBoundSecondsB =
    (nKvCachesPerLinearDeviceB * model.linearFlopsDirty) /
    linearDevice.flopsDirty;
  const linearSecondsA = Math.max(
    linearMemoryBoundSeconds,
    linearComputeBoundSecondsA
  );
  const linearSecondsB = Math.max(
    linearMemoryBoundSeconds,
    linearComputeBoundSecondsB
  );
  const linearBoundReasonA =
    linearMemoryBoundSeconds > linearComputeBoundSecondsA
      ? "memory"
      : "compute";
  const linearBoundReasonB =
    linearMemoryBoundSeconds > linearComputeBoundSecondsB
      ? "memory"
      : "compute";

  const kvCacheSecondsA =
    (nKvCachesA * model.kvCacheWholeBytes) / kvDevice.bytesPerSecond;
  const kvCacheSecondsB =
    (nKvCachesB * model.kvCacheWholeBytes) / kvDevice.bytesPerSecond;

  const totalSeconds =
    Math.max(linearSecondsA, kvCacheSecondsB) +
    Math.max(linearSecondsB, kvCacheSecondsA);
  const tokensPerSecondSequential = 1 / totalSeconds;
  const tokensPerSecondParallel = nKvCachesTotal / totalSeconds;

  return {
    nKvCachesTotal,
    nKvCachesA,
    nKvCachesB,
    linearMemoryBoundSeconds,
    nKvCachesPerLinearDeviceA,
    nKvCachesPerLinearDeviceB,
    linearComputeBoundSecondsA,
    linearComputeBoundSecondsB,
    linearSecondsA,
    linearSecondsB,
    linearBoundReasonA,
    linearBoundReasonB,
    kvCacheSecondsA,
    kvCacheSecondsB,
    totalSeconds,
    tokensPerSecondSequential,
    tokensPerSecondParallel,
  };
}

const llamas: { [size: string]: ModelArchitecture } = {
  "7b": {
    nContext: 512,
    nEmbedding: 4096,
    nHeads: 32,
    nLayers: 32,
    roughParams: 7e9,
  },
  "13b": {
    nContext: 512,
    nEmbedding: 5120,
    nHeads: 40,
    nLayers: 40,
    roughParams: 13e9,
  },
  "33b": {
    nContext: 512,
    nEmbedding: 6656,
    nHeads: 52,
    nLayers: 60,
    roughParams: 33e9,
  },
  "65b": {
    nContext: 512,
    nEmbedding: 8192,
    nHeads: 64,
    nLayers: 80,
    roughParams: 65e9,
  },
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

const configurations: { name: string; calculate: (model: Model) => {} }[] = [
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
      calculateScenarioSplitLinearAndKvAlternating({
        model,
        kvDevice: devices.gpu,
        linearDevice: devices.cpu,
      }),
  },
  {
    name: "GPU linear, CPU KV alternating",
    calculate: (model) =>
      calculateScenarioSplitLinearAndKvAlternating({
        model,
        kvDevice: devices.cpu,
        linearDevice: devices.gpu,
      }),
  },
  {
    name: "GPU linear, CPU KV interleaved",
    calculate: (model) =>
      calculateScenarioSplitLinearAndKvInterleaved({
        model,
        kvDevice: devices.cpu,
        kvDeviceCount: 1,
        linearDevice: devices.gpu,
        linearDeviceCount: 1,
      }),
  },
  {
    name: "Dual GPU alternating",
    calculate: (model) =>
      calculateScenarioSplitLinearAndKvAlternating({
        model,
        kvDevice: devices.gpu,
        linearDevice: devices.gpu,
      }),
  },
  {
    name: "Dual GPU interleaved",
    calculate: (model) =>
      calculateScenarioSplitLinearAndKvInterleaved({
        model,
        kvDevice: devices.gpu,
        kvDeviceCount: 1,
        linearDevice: devices.gpu,
        linearDeviceCount: 1,
      }),
  },
];

const output: { [model: string]: any } = {};
for (const [llamaSize, llama] of Object.entries(llamas)) {
  output[llamaSize] = {};
  for (const bytesPerParam of [0.5, 1, 2, 4]) {
    const model = new Model(llama, bytesPerParam);
    const outputsThisModel: {}[] = [];
    output[llamaSize][bytesPerParam] = outputsThisModel;
    for (const configuration of configurations) {
      try {
        outputsThisModel.push({
          configuration: configuration.name,
          ...configuration.calculate(model),
        });
      } catch (e) {
        outputsThisModel.push({
          configuration: configuration.name,
          error: (e as any).message,
        });
      }
    }
  }
}
console.log(JSON.stringify(output, null, 2));
