use rand_distr::{Distribution, Normal};
use std::fmt::{self, Debug};

struct NeuralNetworkLayer {
    dim_in: u32,
    dim_out: u32,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    activation: fn(f64) -> f64,
    derivative: fn(f64) -> f64,
}

impl NeuralNetworkLayer {
    fn new(
        dim_in: u32,
        dim_out: u32,
        activation: fn(f64) -> f64,
        derivative: fn(f64) -> f64,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        Self {
            dim_in: dim_in,
            dim_out: dim_out,
            a: (0..dim_out)
                .map(|_| {
                    (0..dim_in)
                        .collect::<Vec<u32>>()
                        .iter()
                        .map(|_| normal.sample(&mut rng))
                        .collect()
                })
                .collect(),
            b: (0..dim_in)
                .collect::<Vec<u32>>()
                .iter()
                .map(|_| normal.sample(&mut rng))
                .collect(),
            activation: activation,
            derivative: derivative,
        }
    }

    fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let linear: Vec<f64> = (0..self.dim_out as usize)
            .collect::<Vec<_>>()
            .iter()
            .map(|i| {
                &input
                    .iter()
                    .enumerate()
                    .map(|(j, x)| self.a[*i][j] * x)
                    .fold(0.0, |acc, y| acc + y)
                    + self.b[*i]
            })
            .collect();

        (
            linear.iter().map(|x| (self.activation)(*x)).collect(),
            linear.iter().map(|x| (self.derivative)(*x)).collect(),
        )
    }

    fn back(&self, error: &[f64], derivatives: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
        let partial: Vec<Vec<f64>> = (self.a)
            .iter()
            .enumerate()
            .map(|(j, row)| row.iter().map(|x| x * error[j]).collect())
            .collect();
        let new_error = (0..(self.dim_in) as usize)
            .collect::<Vec<_>>()
            .iter()
            .map(|i| {
                derivatives
                    .iter()
                    .enumerate()
                    .map(|(j, df)| df * partial[j][*i])
                    .fold(0.0, |acc, x| acc + x)
            })
            .collect();
        (partial, new_error)
    }
}

impl Debug for NeuralNetworkLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Neural Network Layer")
            .field("input dimensions", &self.dim_in)
            .field("output dimensions", &self.dim_out)
            .field("layer weights", &self.a)
            .field("layer bias", &self.b)
            .finish()
    }
}

#[test]
fn test_neural_network_layer_forward() {
    let relu = |x| if x > 0.0 { x } else { 0.0 };
    let step = |x| if x > 0.0 { 1.0 } else { 0.0 };
    let mut nn = NeuralNetworkLayer::new(5, 3, relu, step);
    println!("{nn:?}");
    nn.a = vec![
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
        vec![-1.0, -1.0, -1.0, -1.0, -1.0],
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
    ];
    nn.b = vec![1.0, 1.0, 1.0];
    let result = nn.forward(&[5.0, 6.0, 7.0, 8.0, 9.0]);
    assert_eq!(result.0, vec![36.0, 0.0, 36.0]);
}

#[test]
fn test_neural_network_layer_back() {
    let relu = |x| if x > 0.0 { x } else { 0.0 };
    let step = |x| if x > 0.0 { 1.0 } else { 0.0 };
    let mut nn = NeuralNetworkLayer::new(5, 3, relu, step);
    println!("{nn:?}");
    nn.a = vec![
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
        vec![-1.0, -1.0, -1.0, -1.0, -1.0],
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
    ];
    nn.b = vec![1.0, 1.0, 1.0];
    let result = nn.forward(&[5.0, 6.0, 7.0, 8.0, 9.0]);
    let back_result = nn.back(&[1.0, 1.0, 1.0], &result.1);
    assert_eq!(back_result.0.len() as u32, nn.dim_out);
    assert_eq!(back_result.0[0].len() as u32, nn.dim_in);
    assert_eq!(back_result.1.len() as u32, nn.dim_in);
}
