#![feature(iter_array_chunks)]

mod loader;
mod model;
mod prediction;
mod tree;
mod ui;
mod vocab;

use std::{
    fs::File,
    io::{BufWriter, Write},
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
    let n_cache = 128; // HACK small for testing CUDA layers with little VRAM
    let (mut model, tokenizer, vocab_embeddings) = Model::load_gptq(path, params, n_cache)?;

    let input_str = "The average cost of a wedding is";
    let input_encoding = tokenizer
        .encode(input_str, true)
        .map_err(|err| anyhow!(err))?;

    let mut prediction_path = vec![];
    let mut next_token_id = input_encoding.get_ids()[0];
    let mut i_input = 0;
    for i in 1..n_cache {
        i_input += 1;

        if let Some(next_token_str) = tokenizer.id_to_token(next_token_id) {
            print!("{}", next_token_str);
            std::io::stdout().flush().unwrap();
        }

        prediction_path.push(model.next_i());
        let hidden_in = vocab_embeddings.get_embedding(next_token_id.try_into().unwrap());
        let logits = model.predict(hidden_in, &prediction_path);

        let argmax_logits = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        if i == 32 {
            prediction_path.clear();
            model.retain(&[]);
            next_token_id = input_encoding.get_ids()[0];
            i_input = 0;
            assert_eq!(model.next_i(), 0);
            println!();
            continue;
        }

        if i_input < input_encoding.len() {
            next_token_id = input_encoding.get_ids()[i_input].try_into().unwrap();
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

#[cfg(test)]
mod tests {
    use std::{fs::File, path::Path};

    use anyhow::Result;
    use cpp_stuff_nice::LlamaHyperparams;
    use memmap2::MmapOptions;
    use rand::Rng;

    use crate::loader::GPTQLlamaLoader;

    const MODEL_PATH: &str = "/home/philippe/Documents/llama2/Llama-2-7B-GPTQ/"; // HACK

    #[test]
    fn test_cuda_layer() -> Result<()> {
        let params = LlamaHyperparams {
            n_hidden: 4096,
            n_context: 4096,
            n_heads: 32,
            n_ff: 11008,
            n_vocab: 32000,
            n_layers: 32,
            gptq_block_size: 128,
        };
        let n_cache = 128;
        let n_hidden = usize::try_from(params.n_hidden).unwrap();

        let weights_path = Path::new(MODEL_PATH).join("gptq_model-4bit-128g.safetensors");

        let weights_buffer = {
            let file = File::open(weights_path)?;
            unsafe { MmapOptions::new().map(&file)? }
        };
        let loader = GPTQLlamaLoader::new(&weights_buffer, params)?;

        let layer_weights = loader.load_layer(0)?;
        let mut baseline_layer =
            cpp_stuff_nice::baseline::create_llama_layer_gptq(&layer_weights, params, n_cache);
        let mut cuda_layer =
            cpp_stuff_nice::cuda::create_llama_layer_gptq(&layer_weights, params, n_cache);

        let mut rng = rand::thread_rng();

        for _ in 0..4 {
            let hidden_in = (0..n_hidden).map(|_| rng.gen()).collect::<Vec<_>>();
            let mut final_out_baseline = vec![0.0; n_hidden];
            let mut final_out_cuda = vec![0.0; n_hidden];

            baseline_layer.forward(
                &hidden_in,
                &mut final_out_baseline,
                &[baseline_layer.next_i()],
            );
            cuda_layer.forward(&hidden_in, &mut final_out_cuda, &[cuda_layer.next_i()]);

            let tolerance = 0.0001;
            let mut all_close_enough = true;
            for (i, (a, b)) in final_out_baseline
                .iter()
                .zip(final_out_cuda.iter())
                .enumerate()
            {
                let close_enough = (a - b).abs() < tolerance;
                if !close_enough {
                    all_close_enough = false;
                    println!("{}: {} != {} (tolerance {})", i, a, b, tolerance)
                }
            }
            assert!(all_close_enough);
        }

        Ok(())
    }
}
