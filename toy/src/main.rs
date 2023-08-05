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
    panic::Location,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Result};
use cpp_stuff_nice::LlamaHyperparams;
use loader::GPTQLlamaLoader;
use memmap2::MmapOptions;
use model::Model;
use tokenizers::Tokenizer;
use tracing_subscriber::prelude::*;

use crate::{prediction::prediction_thread_main, tree::InferenceTree};

fn test_thing(path: impl AsRef<Path>) -> Result<()> {
    let params = LlamaHyperparams {
        n_hidden: 4096,
        n_context: 4096,
        n_heads: 32,
        n_ff: 11008,
        n_vocab: 32000,
        n_layers: 32,
        gptq_block_size: 128,
    };
    let (mut model, tokenizer, vocab_embeddings) = Model::load_gptq(path, params)?;

    let input_str = "function isPrime(x) {";
    let input_encoding = tokenizer
        .encode(input_str, true)
        .map_err(|err| anyhow!(err))?;

    let mut prediction_path = vec![];
    let mut next_token_id = input_encoding.get_ids()[0];
    for i in 1..(input_encoding.len() + 100) {
        let next_token_str = tokenizer
            .decode(vec![next_token_id], false)
            .map_err(|err| anyhow!(err))?;
        println!("next_token {}: {}", next_token_id, next_token_str);

        prediction_path.push(model.next_i());
        let hidden_in = vocab_embeddings.get_embedding(next_token_id.try_into().unwrap());
        let logits = model.predict(hidden_in, &prediction_path);

        let argmax_logits = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        let argmax_logits_str = tokenizer
            .decode(vec![argmax_logits.try_into().unwrap()], false)
            .map_err(|err| anyhow!(err))?;
        println!("argmax_logits {}: {}", argmax_logits, argmax_logits_str);

        if i < input_encoding.len() {
            next_token_id = input_encoding.get_ids()[i].try_into().unwrap();
        } else {
            next_token_id = argmax_logits.try_into().unwrap();
        }
    }

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
