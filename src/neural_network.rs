use rand_distr::{Distribution, Normal};
use std::{
    array,
    fmt::{self, Debug},
};

trait Activation {
    fn activation(&self) -> fn(&[f64]) -> Vec<f64>;
}

trait Derivative {
    fn derivative(&self) -> fn(&[f64], &[f64]) -> Vec<f64>;
}

trait Layer {
    fn forward(&self, input: &[f64]) -> Vec<f64>;
    fn back(&self, output: &[f64], error: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>);
}

#[derive(Clone, Copy)]
enum ActivationFunction {
    ReLU,
}

impl Activation for ActivationFunction {
    fn activation(&self) -> fn(&[f64]) -> Vec<f64> {
        match self {
            Self::ReLU => |x| {
                x.iter()
                    .map(|xi| if *xi > 0.0 { *xi } else { 0.0 })
                    .collect()
            },
        }
    }
}

impl Derivative for ActivationFunction {
    fn derivative(&self) -> fn(&[f64], &[f64]) -> Vec<f64> {
        match self {
            Self::ReLU => |x, y| {
                x.iter()
                    .zip(y.iter())
                    .map(|(xi, yi)| if *xi > 0.0 { *yi } else { 0.0 })
                    .collect()
            },
        }
    }
}

#[derive(Clone, Copy)]
enum LossFunction {
    CrossEntropy,
}

impl LossFunction {
    fn loss(&self) -> fn(&[f64], &[f64]) -> f64 {
        match self {
            Self::CrossEntropy => |x, y| {
                x.iter()
                    .zip(y.iter())
                    .map(|(xi, yi)| xi * yi.ln())
                    .fold(0.0, |acc, z| acc + z)
                    / (x.len() as f64)
            },
        }
    }
}
impl Activation for LossFunction {
    fn activation(&self) -> fn(&[f64]) -> Vec<f64> {
        match self {
            Self::CrossEntropy => |x: &[f64]| {
                let denom = x.iter().map(|y| y.exp()).fold(0.0, |acc, y| acc + y);
                x.iter().map(|y| y.exp() / denom).collect()
            },
        }
    }
}

impl Derivative for LossFunction {
    fn derivative(&self) -> fn(&[f64], &[f64]) -> Vec<f64> {
        match self {
            Self::CrossEntropy => |x, y| x.iter().zip(y.iter()).map(|(xi, yi)| *yi - *xi).collect(),
        }
    }
}
#[derive(Clone, Copy)]
enum Function {
    ReLU,
    CrossEntropy,
}

impl Activation for Function {
    fn activation(&self) -> fn(&[f64]) -> Vec<f64> {
        match self {
            Self::CrossEntropy => |x: &[f64]| {
                let denom = x.iter().map(|y| y.exp()).fold(0.0, |acc, y| acc + y);
                x.iter().map(|y| y.exp() / denom).collect()
            },
            Self::ReLU => |x| {
                x.iter()
                    .map(|xi| if *xi > 0.0 { *xi } else { 0.0 })
                    .collect()
            },
        }
    }
}

impl Derivative for Function {
    fn derivative(&self) -> fn(&[f64], &[f64]) -> Vec<f64> {
        match self {
            Self::CrossEntropy => |x, y| x.iter().zip(y.iter()).map(|(xi, yi)| *yi - *xi).collect(),
            Self::ReLU => |x, y| {
                x.iter()
                    .zip(y.iter())
                    .map(|(xi, yi)| if *xi > 0.0 { *yi } else { 0.0 })
                    .collect()
            },
        }
    }
}

struct NeuralNetworkLayer {
    dim_in: u32,
    dim_out: u32,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    cap: Function,
}

impl NeuralNetworkLayer {
    fn new(dim_in: u32, dim_out: u32, cap: Function) -> Self {
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
    fn forward(&self, input: &[f64]) -> Vec<f64> {
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

        (self.cap).activation()(&linear[..])
    }

    fn back(&self, output: &[f64], error: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let res = self.cap.derivative()(error, output);
        let partial: Vec<Vec<f64>> = (self.a)
            .iter()
            .enumerate()
            .map(|(j, row)| row.iter().map(|x| x * res[j]).collect())
            .collect();
        let bias: Vec<f64> = self.b.iter().enumerate().map(|(i, x)| x * res[i]).collect();
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
    let mut nn = NeuralNetworkLayer::new(5, 3, Function::ReLU);
    println!("{nn:?}");
    nn.a = vec![
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
        vec![-1.0, -1.0, -1.0, -1.0, -1.0],
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
    ];
    nn.b = vec![1.0, 1.0, 1.0];
    let result = nn.forward(&[5.0, 6.0, 7.0, 8.0, 9.0]);
    assert_eq!(result, vec![36.0, 0.0, 36.0]);
}

#[test]
fn test_neural_network_layer_back() {
    let mut nn = NeuralNetworkLayer::new(5, 3, Function::ReLU);
    println!("{nn:?}");
    nn.a = vec![
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
        vec![-1.0, -1.0, -1.0, -1.0, -1.0],
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
    ];
    nn.b = vec![1.0, 1.0, 1.0];
    let result = nn.forward(&[5.0, 6.0, 7.0, 8.0, 9.0]);
    let back_result = nn.back(&result[..], &[1.0, 1.0, 1.0]);
    assert_eq!(back_result.0.len() as u32, nn.dim_out);
    assert_eq!(back_result.0[0].len() as u32, nn.dim_in);
    assert_eq!(back_result.1.len() as u32, nn.dim_out);
    assert_eq!(back_result.2.len() as u32, nn.dim_in);
}

fn linear_nn(dim: &[u32], activation: Function, loss: Function) -> Vec<NeuralNetworkLayer> {
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

#[test]
fn test_linear_nn() {
    let linear = linear_nn(&[16, 8, 4, 2], Function::ReLU, Function::CrossEntropy);
    let input: Vec<f64> = (0..16)
        .collect::<Vec<_>>()
        .iter()
        .map(|x| *x as f64)
        .collect();
    let result = linear[2].forward(&linear[1].forward(&linear[0].forward(&input[..])[..])[..]);
    println!("{result:?}");
    assert_eq!(linear.len(), 3);
    assert_eq!(linear[0].dim_in, 16);
    assert_eq!(linear[2].dim_out, 2);
    assert_eq!(linear[0].dim_out, linear[1].dim_in);
    assert_eq!(linear[1].dim_out, linear[2].dim_in);
}
