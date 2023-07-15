#![feature(iter_array_chunks)]

mod model;
mod prediction;
mod tree;
mod ui;
mod vocab;

use std::{
    fs::File,
    io::BufWriter,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use tracing_subscriber::prelude::*;

use crate::{prediction::prediction_thread_main, tree::InferenceTree};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        println!("Usage: toy <model_path>");
        return;
    }
    let model_path = PathBuf::from(&args[1]);

    let trace_writer = BufWriter::new(File::create("trace.json").unwrap());
    let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .writer(trace_writer)
        .include_args(true)
        .build();
    tracing_subscriber::registry().with(chrome_layer).init();

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
    .run();

    guard.flush();

    // let input_text = "5 JavaScript edge cases that broke prod\n";
    // let input_tokens_ids: Vec<usize> = vocab
    //     .tokenize(input_text, true)
    //     .unwrap()
    //     .into_iter()
    //     .map(|(_, id)| id.try_into().unwrap())
    //     .collect();

    // let mut last_token_id = input_tokens_ids.first().unwrap().clone();
    // let mut print_buffer = TokenUtf8Buffer::new();
    // for i_context in 0..model.n_context {
    //     let span = span!(tracing::Level::INFO, "token processing", i_context);
    //     let _enter = span.enter();

    //     if i_context % 10 == 0 {
    //         guard.flush();
    //     }

    //     let input_token_id = if i_context < input_tokens_ids.len() {
    //         input_tokens_ids[i_context].clone()
    //     } else {
    //         last_token_id
    //     };

    //     if let Some(s) = print_buffer.push(vocab.token(input_token_id)) {
    //         print!("{}", s);
    //         io::stdout().flush().unwrap();
    //     }

    //     let hidden_in = vocab_embeddings.get_embedding(input_token_id.try_into().unwrap());
    //     prediction_path.push(model.next_i());
    //     let final_out = model.predict(&hidden_in, &prediction_path);
    //     let token_id = final_out
    //         .iter()
    //         .enumerate()
    //         .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    //         .unwrap()
    //         .0;
    //     last_token_id = token_id;
    // }
    // println!();

    // guard.flush();
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    const MODEL_PATH: &str = "/home/philippe/Documents/llama/7B/ggml-model-f16.bin"; // HACK

    #[test]
    fn test_cuda_final_layer() {
        let mut loader = cpp_stuff_nice::SimpleLlamaModelLoader::new(MODEL_PATH);
        let n_hidden = usize::try_from(loader.n_hidden()).unwrap();
        let n_vocab = usize::try_from(loader.n_vocab()).unwrap();

        let mut baseline_layer = cpp_stuff_nice::baseline::create_llama_final_layer(&mut loader);
        let mut cuda_layer = cpp_stuff_nice::cuda::create_llama_final_layer(&mut loader);

        let mut rng = rand::thread_rng();

        let mut hidden_in = (0..n_hidden).map(|_| rng.gen()).collect::<Vec<_>>();
        let mut final_out_baseline = vec![0.0; n_vocab];
        let mut final_out_cuda = vec![0.0; n_vocab];

        baseline_layer.forward(
            &hidden_in,
            &mut final_out_baseline,
            &[baseline_layer.next_i()],
        );
        cuda_layer.forward(&hidden_in, &mut final_out_cuda, &[cuda_layer.next_i()]);

        let tolerance = 0.1;
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
}
