use crate::neural_network::core::{linear_transform, outer, Layer};

pub struct Linear {
    dim_in: usize,
    dim_out: usize,
    weights: Vec<Vec<f32>>,
}

impl Layer for Linear {
    fn dim_in(&self) -> usize {
        self.dim_in
    }
    fn dim_out(&self) -> usize {
        self.dim_out
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        linear_transform(&self.weights, input, None)
    }

    fn backward(
        &self,
        input: &[f32],
        error: &[f32],
    ) -> (Vec<f32>, Option<Vec<Vec<f32>>>, Option<Vec<f32>>) {
        let weight_error = outer(error, input);
        (
            weight_error
                .iter()
                .map(|x| x.clone())
                .reduce(|acc, x| acc.iter().zip(x.iter()).map(|(i, j)| i + j).collect())
                .unwrap(),
            Some(weight_error),
            None,
        )
    }
}
