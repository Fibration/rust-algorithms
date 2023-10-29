use rand_distr::{Distribution, Normal};
use std::{
    array,
    fmt::{self, Debug},
};

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

    fn back(&self, error: &[f64], derivatives: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let partial: Vec<Vec<f64>> = (self.a)
            .iter()
            .enumerate()
            .map(|(j, row)| row.iter().map(|x| x * derivatives[j] * error[j]).collect())
            .collect();
        let bias: Vec<f64> = self
            .b
            .iter()
            .enumerate()
            .map(|(i, x)| x * derivatives[i] * error[i])
            .collect();
        let new_error = (0..(self.dim_in) as usize)
            .collect::<Vec<_>>()
            .iter()
            .map(|i| {
                (0..(self.dim_out) as usize)
                    .collect::<Vec<_>>()
                    .iter()
                    .map(|j| partial[*j][*i])
                    .fold(0.0, |acc, x| acc + x)
            })
            .collect();
        (partial, bias, new_error)
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
    assert_eq!(back_result.1.len() as u32, nn.dim_out);
    assert_eq!(back_result.2.len() as u32, nn.dim_in);
}

enum LossFunction {
    CrossEntropy,
}

impl LossFunction {
    fn activation(&self) -> fn(&[f64]) -> Vec<f64> {
        match self {
            Self::CrossEntropy => |x: &[f64]| {
                let denom = x.iter().map(|y| y.exp()).fold(0.0, |acc, y| acc + y);
                x.iter().map(|y| y.exp() / denom).collect()
            },
        }
    }

    fn derivative(&self) -> fn(&[f64], &[f64]) -> Vec<f64> {
        match self {
            Self::CrossEntropy => |x, y| x.iter().zip(y.iter()).map(|(xi, yi)| xi - yi).collect(),
        }
    }
}

// consider varying the final activation function, i.e. softmax
fn linear_nn(dim: &[u32]) -> Vec<NeuralNetworkLayer> {
    dim[..(dim.len() - 1)]
        .iter()
        .zip(dim[1..].iter())
        .map(|(m, n)| {
            NeuralNetworkLayer::new(
                *m,
                *n,
                |x| if x > 0.0 { x } else { 0.0 },
                |x| if x > 0.0 { 1.0 } else { 0.0 },
            )
        })
        .collect()
}

#[test]
fn test_linear_nn() {
    let linear = linear_nn(&[16, 8, 4, 2]);
    let input: Vec<f64> = (0..16)
        .collect::<Vec<_>>()
        .iter()
        .map(|x| *x as f64)
        .collect();
    let result = linear[2]
        .forward(&linear[1].forward(&linear[0].forward(&input[..]).0[..]).0[..])
        .0;
    println!("{result:?}");
    assert_eq!(linear.len(), 3);
    assert_eq!(linear[0].dim_in, 16);
    assert_eq!(linear[2].dim_out, 2);
    assert_eq!(linear[0].dim_out, linear[1].dim_in);
    assert_eq!(linear[1].dim_out, linear[2].dim_in);
}
