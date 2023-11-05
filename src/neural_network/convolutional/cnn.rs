use crate::neural_network::core::{stack, unstack, Layer};

use super::ConvolutionLayer;

struct CNNLayer {
    num_in: usize,
    filters: Vec<ConvolutionLayer>,
}

impl Layer for CNNLayer {
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
        (todo!(), todo!(), todo!())
    }
}
