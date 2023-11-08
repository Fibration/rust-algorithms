use std::collections::HashMap;

use crate::neural_network::core::{stack, unstack, Function, Layer};

use super::ConvolutionLayer;

#[derive(Clone)]
struct CNNLayer {
    num_in: usize,
    filters: Vec<ConvolutionLayer>,
}

impl Layer for CNNLayer {
    fn cap(&self) -> Function {
        self.filters[0].cap
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let inputs = stack(input, (self.num_in, input.len() / self.num_in));
        let mut outputs = Vec::new();
        for input_piece in inputs {
            for filter in &self.filters {
                outputs.push(filter.forward(&input_piece))
            }
        }
        unstack(&outputs)
    }

    fn back(&self, input: &[f64], error: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let inputs = stack(input, (self.num_in, input.len() / self.num_in));
        let num_out = self.num_in * self.filters.len();
        let errors = stack(error, (num_out, error.len() / num_out));

        let mut new_error = Vec::new();
        let mut filter_deltas = HashMap::new();
        let mut bias_result = Vec::new();

        for i in 0..self.filters.len() {
            bias_result.push(0.0);
            filter_deltas.insert(i, Vec::new());
        }
        for input_piece in inputs {
            let mut error_piece_contributors = Vec::new();
            for filter in self.filters.iter().enumerate() {
                for error in &errors {
                    let back_result = filter.1.back(&input_piece, &error);
                    let current = filter_deltas.get_mut(&filter.0).unwrap();
                    current.push(unstack(&back_result.0));
                    bias_result[filter.0] += back_result.1[0];
                    error_piece_contributors.push(back_result.2);
                }
            }
            // every contributor to an input has to be averaged so input:new_error is 1:1
            let mut error_piece = Vec::new();
            for i in 0..error_piece_contributors[0].len() {
                error_piece.push(
                    error_piece_contributors
                        .iter()
                        .map(|x| x[i])
                        .fold(0.0, |acc, x| acc + x)
                        / error_piece_contributors.len() as f64,
                );
            }
            new_error.push(error_piece);
        }
        // average corrections for each filter
        let mut filter_result = Vec::new();
        for filter in self.filters.iter().enumerate() {
            let current = filter_deltas.get(&filter.0).unwrap();
            let mut filter_average = Vec::new();
            for i in 0..current[0].len() {
                filter_average.push(
                    current.iter().map(|x| x[i]).fold(0.0, |acc, x| acc + x) / current.len() as f64,
                );
            }
            filter_result.push(filter_average);
        }

        (filter_result, bias_result, unstack(&new_error))
    }
}
