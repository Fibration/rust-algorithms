use rand_distr::{Distribution, Normal};

use crate::neural_network::core::{stack, unstack, Activation, Function, Layer};

use super::convolution;

#[derive(Clone)]
pub struct ConvolutionLayer {
    pub dim_in: (usize, usize),
    pub dim_out: (usize, usize),
    pub kernel: (usize, usize),
    pub padding: (usize, usize),
    pub stride: (usize, usize),
    pub a: Vec<Vec<f32>>,
    pub b: Vec<f32>,
    pub cap: Function,
}

impl ConvolutionLayer {
    pub fn new(
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
            b: vec![0.0],
            cap,
        }
    }
}

pub fn pad_right_within(matrix: Vec<Vec<f32>>, padding: (usize, usize)) -> Vec<Vec<f32>> {
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

pub fn pad_around(
    matrix: Vec<Vec<f32>>,
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Vec<Vec<f32>> {
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
    fn cap(&self) -> Function {
        self.cap
    }

    // (row, col)
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let data = stack(input, self.dim_in);
        let result = convolution(&data, &self.a, self.padding, self.stride);
        let output: Vec<f32> = unstack(&result).iter().map(|x| x + self.b[0]).collect();
        (self.cap).activation()(&output[..])
    }

    fn back(&self, input: &[f32], error: &[f32]) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
        let input_matrix = stack(input, self.dim_in);
        let mut error_matrix = stack(error, self.dim_out);
        if self.stride.0 > 1 || self.stride.1 > 1 {
            error_matrix = pad_right_within(error_matrix, (self.stride.0 - 1, self.stride.1 - 1));
        }
        let partial = convolution(&input_matrix, &error_matrix, self.padding, (1, 1));
        let delta_bias =
            error.iter().fold(0.0, |acc, x| acc + x) / (self.dim_out.0 * self.dim_out.1) as f32;
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
