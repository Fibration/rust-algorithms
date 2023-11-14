use crate::neural_network::core::Function;

use super::super::core::{Activation, Derivative, Layer};
use rand_distr::{Distribution, Normal};
use std::fmt::{self, Debug};

#[derive(Clone)]
pub struct NeuralNetworkLayer {
    pub dim_in: u32,
    pub dim_out: u32,
    pub a: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub cap: Function,
}

impl NeuralNetworkLayer {
    pub fn new(dim_in: u32, dim_out: u32, cap: Function) -> Self {
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
            cap: cap,
        }
    }
}
impl Layer for NeuralNetworkLayer {
    fn cap(&self) -> Function {
        self.cap
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let linear: Vec<f64> = (self.a)
            .iter()
            .zip((self.b).iter())
            .map(|(row, b)| {
                row.iter()
                    .zip(input.iter())
                    .map(|(x, y)| x * y)
                    .sum::<f64>()
                    + b
            })
            .collect();
        (self.cap).activation()(&linear[..])
    }

    fn back(&self, output: &[f64], error: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let res = self.cap.derivative()(error, output);
        let partial: Vec<Vec<f64>> = (self.a)
            .iter()
            .zip(res.iter())
            .map(|(row, z)| row.iter().map(|x| x * z).collect())
            .collect();
        let bias: Vec<f64> = self.b.iter().zip(res.iter()).map(|(x, y)| x * y).collect();
        let new_error: Vec<f64> = partial
            .iter()
            .map(|x| x.to_vec())
            .reduce(|acc, x| x.iter().zip(acc.iter()).map(|(y, z)| y + z).collect())
            .unwrap();
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

pub fn linear_nn(dim: &[u32], activation: Function, loss: Function) -> Vec<NeuralNetworkLayer> {
    let mut nn: Vec<NeuralNetworkLayer> = dim[..(dim.len() - 1)]
        .iter()
        .zip(dim[1..].iter())
        .map(|(m, n)| NeuralNetworkLayer::new(*m, *n, activation.clone()))
        .collect();
    nn.pop();
    nn.push(NeuralNetworkLayer::new(
        dim[dim.len() - 2],
        dim[dim.len() - 1],
        loss,
    ));
    nn
}
