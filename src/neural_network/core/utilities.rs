use rand::thread_rng;
use rand_distr::{Distribution, Normal};

pub fn he_initialise(dim_in: usize, dim_out: usize) -> Vec<f32> {
    let factor = (2.0 / (dim_in as f32)).sqrt();
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, factor).unwrap();

    (0..dim_out)
        .collect::<Vec<_>>()
        .iter()
        .map(|_| normal.sample(&mut rng))
        .collect()
}
