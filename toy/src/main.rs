use rand::Rng;
use rand_distr::StandardNormal;

fn do_thing(model_path: &str) {
    let mut rng = rand::thread_rng();

    unsafe {
        let fname_base = std::ffi::CString::new(model_path).unwrap();
        let mut loader = cpp_stuff::cml_SimpleLlamaModelLoader::new(fname_base.as_ptr());
        let n_hidden = usize::try_from((*loader.get_hparams()).n_embd).unwrap();

        let baseline_model = cpp_stuff::cml_cuda_create_llama_layer(&mut loader, 0);
        let cuda_model = cpp_stuff::cml_cuda_create_llama_layer(&mut loader, 0);
        let hip_model = cpp_stuff::cml_hip_create_llama_layer(&mut loader, 0);

        loader.destruct();

        let mut hidden_in: Vec<f32> = (0..n_hidden)
            .map(|_| rng.sample(StandardNormal))
            .collect::<Vec<_>>();
        let mut hidden_out: Vec<f32> = vec![0.0; n_hidden];

        cpp_stuff::cml_simple_transformer_layer_forward(
            baseline_model,
            n_hidden.try_into().unwrap(),
            hidden_in.as_mut_ptr(),
            hidden_out.as_mut_ptr(),
        );
        let hidden_out_baseline = hidden_out.clone();

        cpp_stuff::cml_simple_transformer_layer_forward(
            cuda_model,
            n_hidden.try_into().unwrap(),
            hidden_in.as_mut_ptr(),
            hidden_out.as_mut_ptr(),
        );
        let hidden_out_cuda = hidden_out.clone();

        cpp_stuff::cml_simple_transformer_layer_forward(
            hip_model,
            n_hidden.try_into().unwrap(),
            hidden_in.as_mut_ptr(),
            hidden_out.as_mut_ptr(),
        );
        let hidden_out_hip = hidden_out.clone();

        for i_hidden in [0, 1, n_hidden - 2, n_hidden - 1].into_iter() {
            println!(
                "hidden_out_baseline[{}] = {}",
                i_hidden, hidden_out_baseline[i_hidden]
            );
            println!(
                "hidden_out_cuda[{}] = {}",
                i_hidden, hidden_out_cuda[i_hidden]
            );
            println!(
                "hidden_out_hip[{}] = {}",
                i_hidden, hidden_out_hip[i_hidden]
            );
        }

        cpp_stuff::cml_simple_transformer_layer_delete(baseline_model);
        cpp_stuff::cml_simple_transformer_layer_delete(cuda_model);
        cpp_stuff::cml_simple_transformer_layer_delete(hip_model);
    }
}

fn main() {
    // read model path from first argument
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        println!("Usage: toy <model_path>");
        return;
    }
    let model_path = &args[1];

    println!("Hello, world!");
    do_thing(model_path);
    println!("Goodbye, world!")
}
