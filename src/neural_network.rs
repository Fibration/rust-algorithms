use rand_distr::{num_traits::Float, Distribution, Normal};
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

struct ConvolutionLayer {
    dim_in: (usize, usize),
    dim_out: (usize, usize),
    kernel: (usize, usize),
    padding: (usize, usize),
    stride: (usize, usize),
    a: Vec<Vec<f64>>,
    b: f64,
    cap: Function,
}

impl ConvolutionLayer {
    fn new(
        dim_in: (usize, usize),
        dim_out: (usize, usize),
        kernel: (usize, usize),
        padding: (usize, usize),
        stride: (usize, usize),
        cap: Function,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        ConvolutionLayer {
            dim_in,
            dim_out,
            kernel,
            padding,
            stride,
            a: (0..kernel.0)
                .map(|_| {
                    (0..kernel.1)
                        .collect::<Vec<usize>>()
                        .iter()
                        .map(|_| normal.sample(&mut rng))
                        .collect()
                })
                .collect(),
            b: 0.0,
            cap,
        }
    }
}

#[test]
fn test_stack() {
    assert_eq!(
        stack(&[1.0, 1.0, 1.0, 1.0], (2, 2)),
        Vec::from([[1.0, 1.0], [1.0, 1.0]])
    );
}

fn stack(flat: &[f64], dim: (usize, usize)) -> Vec<Vec<f64>> {
    let mut stacked = Vec::new();
    for i in 0..dim.0 {
        let mut row = Vec::new();
        for j in 0..dim.1 {
            row.push(flat[i * dim.1 + j]);
        }
        stacked.push(row);
    }
    stacked
}

#[test]
fn test_unstack() {
    let matrix: Vec<Vec<f64>> = Vec::from([vec![1.0, 1.0], vec![1.0, 1.0]]);
    assert_eq!(unstack(&matrix), [1.0, 1.0, 1.0, 1.0]);
}

fn unstack(matrix: &[Vec<f64>]) -> Vec<f64> {
    let mut flat = Vec::new();
    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            flat.push(matrix[i][j]);
        }
    }
    flat
}

#[test]
fn test_pad_right_within() {
    let mut matrix = Vec::new();
    matrix.push(vec![1.0, 1.0]);
    matrix.push(vec![1.0, 1.0]);
    let mut expected = Vec::new();
    expected.push(vec![1.0, 0.0, 1.0, 0.0]);
    expected.push(vec![0.0, 0.0, 0.0, 0.0]);
    expected.push(vec![1.0, 0.0, 1.0, 0.0]);
    expected.push(vec![0.0, 0.0, 0.0, 0.0]);
    assert_eq!(pad_right_within(matrix, (1, 1)), expected);
}

fn pad_right_within(matrix: Vec<Vec<f64>>, padding: (usize, usize)) -> Vec<Vec<f64>> {
    let new_row_len = matrix[0].len() * (1 + padding.1);
    let mut empty = Vec::new();
    for _ in 0..(new_row_len) {
        empty.push(0.0);
    }
    let mut padded = Vec::new();
    for i in 0..matrix.len() {
        let mut padded_row = Vec::new();
        for j in 0..matrix[0].len() {
            padded_row.push(matrix[i][j]);
            for _ in 0..padding.1 {
                padded_row.push(0.0);
            }
        }
        padded.push(padded_row);
        for _ in 0..padding.0 {
            padded.push(empty.clone());
        }
    }
    padded
}

#[test]
fn test_pad_around() {
    let mut matrix = Vec::new();
    matrix.push(vec![1.0, 1.0]);
    matrix.push(vec![1.0, 1.0]);
    let mut expected = Vec::new();
    expected.push(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    expected.push(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    expected.push(vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    expected.push(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    expected.push(vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    expected.push(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    expected.push(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(pad_around(matrix, (2, 2), (1, 1)), expected);
}

fn pad_around(
    matrix: Vec<Vec<f64>>,
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Vec<Vec<f64>> {
    let new_row_len = padding.1 * 2 + matrix[0].len() + (matrix[0].len() - 1) * dilation.1;
    let mut empty = Vec::new();
    for _ in 0..(new_row_len) {
        empty.push(0.0);
    }
    let mut padded = Vec::new();
    for _ in 0..padding.0 {
        padded.push(empty.clone());
    }
    for i in 0..matrix.len() {
        let mut padded_row = Vec::new();
        // pad the left
        for _ in 0..padding.1 {
            padded_row.push(0.0);
        }
        for j in 0..(matrix[0].len() - 1) {
            // add column and then dilation
            padded_row.push(matrix[i][j]);
            for _ in 0..dilation.1 {
                padded_row.push(0.0);
            }
        }
        // add final column entry
        padded_row.push(matrix[i][matrix[0].len() - 1]);
        // pad the right
        for _ in 0..padding.1 {
            padded_row.push(0.0);
        }
        padded.push(padded_row);
        // dilate next row unless last row
        if i < matrix.len() - 1 {
            for _ in 0..dilation.0 {
                padded.push(empty.clone());
            }
        }
    }

    for _ in 0..padding.0 {
        padded.push(empty.clone());
    }

    padded
}

impl Layer for ConvolutionLayer {
    // TODO implement stride and padding
    // (row, col)
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let data = stack(input, self.dim_in);
        let result = convolution(&data, &self.a, self.padding, self.stride);
        let output: Vec<f64> = unstack(&result).iter().map(|x| x + self.b).collect();
        (self.cap).activation()(&output[..])
    }

    fn back(&self, input: &[f64], error: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let input_matrix = stack(input, self.dim_in);
        let mut error_matrix = stack(error, self.dim_out);
        if self.stride.0 > 1 || self.stride.1 > 1 {
            error_matrix = pad_right_within(error_matrix, (self.stride.0 - 1, self.stride.1 - 1));
        }
        let partial = convolution(&input_matrix, &error_matrix, self.padding, (1, 1));
        let delta_bias =
            error.iter().fold(0.0, |acc, x| acc + x) / (self.dim_out.0 * self.dim_out.1) as f64;
        let new_error = convolution(
            &pad_around(
                error_matrix,
                (self.kernel.0 - 1, self.kernel.1 - 1),
                (self.stride.0 - 1, self.stride.1 - 1),
            ),
            &self.a,
            (0, 0),
            (1, 1),
        );
        (partial, vec![delta_bias], unstack(&new_error))
    }
}

#[test]
fn test_convlayer_forward() {
    let input = vec![-2.0, -3.0, 2.0, 1.0, 1.5, 2.5, 2.5, 1.5, 1.0, 2.0, 2.0, 1.0];
    let mut conv = ConvolutionLayer::new((3, 4), (2, 3), (2, 2), (0, 0), (1, 1), Function::ReLU);
    conv.a = Vec::new();
    conv.a.push(vec![1.0, 1.0]);
    conv.a.push(vec![1.0, 1.0]);
    let result = conv.forward(&input);
    assert_eq!(result, [0.0, 4.0, 7.0, 7.0, 9.0, 7.0]);
}

#[test]
fn test_convlayer_back() {
    let mut conv = ConvolutionLayer::new((3, 4), (2, 3), (2, 2), (0, 0), (1, 1), Function::ReLU);
    conv.a = Vec::new();
    conv.a.push(vec![1.0, 1.0]);
    conv.a.push(vec![1.0, 1.0]);
    let input = vec![-2.0, -3.0, 2.0, 1.0, 1.5, 2.5, 2.5, 1.5, 1.0, 2.0, 2.0, 1.0];
    let output = vec![0.0, 4.0, 7.0, 7.0, 9.0, 7.0];
    let error = vec![0.0, 1.0, 2.0, 1.0, 1.0, 1.0];
    let back = conv.back(&input, &error);
    let expected_partial = stack(&[7.5, 10.5,12.5, 10.5], (2,2));
    let expected_error = [0.0, 1.0, 3.0, 2.0, 1.0, 3.0, 5.0, 3.0, 1.0, 2.0, 2.0, 1.0];
    assert_eq!(back.0, expected_partial);
    assert_eq!(back.1, [1.0]);
    assert_eq!(back.2, expected_error);
}

#[test]
fn test_convolution() {
    let mut input = Vec::new();
    input.push(vec![1.0, 2.0, 2.0, 1.0]);
    input.push(vec![1.5, 2.5, 2.5, 1.5]);
    input.push(vec![1.5, 2.5, 2.5, 1.5]);
    input.push(vec![1.0, 2.0, 2.0, 1.0]);
    let mut filter = Vec::new();
    filter.push(vec![1.0, 1.0]);
    filter.push(vec![1.0, 1.0]);
    let output = convolution(&input, &filter, (1, 1), (2, 2));
    assert_eq!(output.len(), 3);
    assert_eq!(output[0].len(), 3);
    assert_eq!(output[0], [1.0, 4.0, 1.0]);
    assert_eq!(output[1], [3.0, 10.0, 3.0]);
    assert_eq!(output[2], [1.0, 4.0, 1.0]);
}

fn convolution(
    data: &[Vec<f64>],
    filter: &[Vec<f64>],
    padding: (usize, usize),
    stride: (usize, usize),
) -> Vec<Vec<f64>> {
    let mut padded = Vec::new();
    for _ in 0..padding.0 {
        let mut row = Vec::new();
        for _ in 0..(data[0].len() + 2 * padding.1) {
            row.push(0.0);
        }
        padded.push(row);
    }
    for i in 0..data.len() {
        let mut row = Vec::new();
        for _ in 0..padding.1 {
            row.push(0.0);
        }
        for j in 0..data[i].len() {
            row.push(data[i][j]);
        }
        for _ in 0..padding.1 {
            row.push(0.0);
        }
        padded.push(row);
    }
    for _ in 0..padding.0 {
        let mut row = Vec::new();
        for _ in 0..(data[0].len() + 2 * padding.1) {
            row.push(0.0);
        }
        padded.push(row);
    }
    let mut output = Vec::new();
    for i in 0..((padded.len() - filter.len() + stride.0) / stride.0) {
        let mut row = Vec::new();
        for j in 0..(padded[0].len() - filter[0].len() + stride.1) / stride.1 {
            let mut convolution = 0.0;
            for k in 0..filter.len() {
                for l in 0..filter[0].len() {
                    convolution += filter[k][l] * padded[k + i * stride.0][l + j * stride.1];
                }
            }
            row.push(convolution);
        }
        output.push(row);
    }
    output
}
