use rand::Rng;
use rand_distr::StandardNormal;

fn do_thing(model_path: &str) {
    let mut rng = rand::thread_rng();

    let mut loader = cpp_stuff_nice::SimpleLlamaModelLoader::new(model_path);
    let n_hidden = usize::try_from(loader.n_hidden()).unwrap();

    let mut baseline_model = cpp_stuff_nice::baseline::create_llama_layer(&mut loader, 0);
    let mut cuda_model = cpp_stuff_nice::cuda::create_llama_layer(&mut loader, 0);
    let mut hip_model = cpp_stuff_nice::hip::create_llama_layer(&mut loader, 0);
    drop(loader);

    let hidden_in: Vec<f32> = (0..n_hidden)
        .map(|_| rng.sample(StandardNormal))
        .collect::<Vec<_>>();
    let mut hidden_out: Vec<f32> = vec![0.0; n_hidden];

    baseline_model.forward(&hidden_in, &mut hidden_out);
    let hidden_out_baseline = hidden_out.clone();

    cuda_model.forward(&hidden_in, &mut hidden_out);
    let hidden_out_cuda = hidden_out.clone();

    hip_model.forward(&hidden_in, &mut hidden_out);
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
}

fn main() {
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
