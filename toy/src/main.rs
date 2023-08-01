#![feature(iter_array_chunks)]

mod loader;
mod model;
mod prediction;
mod tree;
mod ui;
mod vocab;

use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Result};
use cpp_stuff_nice::LlamaHyperparams;
use loader::GPTQLlamaLoader;
use memmap2::MmapOptions;
use tokenizers::Tokenizer;
use tracing_subscriber::prelude::*;

use crate::{prediction::prediction_thread_main, tree::InferenceTree};

fn test_thing(model_path: impl AsRef<Path>) -> Result<()> {
    let model_path = model_path.as_ref();
    let weights_path = model_path.join("gptq_model-4bit-128g.safetensors");
    let tokenizer_path = model_path.join("tokenizer.json");

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|err| anyhow!(err))?;

    let params = LlamaHyperparams {
        n_hidden: 4096,
        n_context: 4096,
        n_heads: 32,
        n_ff: 11008,
        n_vocab: 32000,
        n_layers: 32,
        gptq_block_size: 128,
    };
    let n_cache = params.n_context.try_into().unwrap();

    let weights_buffer = {
        let file = File::open(weights_path)?;
        unsafe { MmapOptions::new().map(&file)? }
    };
    let loader = GPTQLlamaLoader::new(&weights_buffer, params)?;

    let vocab_embeddings = loader.load_vocab_embeddings()?;

    let layers = (0..(params.n_layers as usize))
        .map(|layer_index| {
            let layer_weights = loader.load_layer(layer_index)?;
            Ok(cpp_stuff_nice::baseline::create_llama_layer_gptq(
                &layer_weights,
                params,
                n_cache,
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    let final_layer = {
        let final_layer_weights = loader.load_final_layer()?;
        cpp_stuff_nice::baseline::create_llama_final_layer(&final_layer_weights, params)
    };

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        return Err(anyhow::anyhow!("Usage: toy <model_path>"));
    }
    let model_path = PathBuf::from(&args[1]);
    if !model_path.is_dir() {
        return Err(anyhow::anyhow!(
            "{} is not a directory",
            model_path.display()
        ));
    }

    let trace_writer = BufWriter::new(File::create("trace.json").unwrap());
    let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .writer(trace_writer)
        .include_args(true)
        .build();
    tracing_subscriber::registry().with(chrome_layer).init();

    test_thing(model_path).unwrap();

    return Ok(());

    let bos_token_id = 1;
    let inference_tree = Arc::new(Mutex::new(InferenceTree::new(bos_token_id)));
    let focused_path = Arc::new(Mutex::new(vec![bos_token_id]));

    {
        let inference_tree = inference_tree.clone();
        let focused_path = focused_path.clone();
        std::thread::spawn(move || {
            prediction_thread_main(model_path, inference_tree, focused_path);
        });
    }

    nannou::app::Builder::new_async(move |app| {
        Box::new(async { ui::UIModel::new(app, inference_tree, focused_path) })
    })
    .update(|app, model, update| model.update(app, update))
    .event(|app, model, event| model.event(app, event))
    .run();

    guard.flush();

    Ok(())
}
