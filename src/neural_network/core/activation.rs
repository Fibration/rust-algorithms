use super::{linear_transform, outer};

struct ReLU {
    dim_in: usize,
    dim_out: usize,
}

impl ReLU {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.dim_in);
        input
            .iter()
            .map(|x| if *x > 0.0 { *x } else { 0.0 })
            .collect()
    }

    fn backward(
        &self,
        input: &[f32],
        error: &[f32],
    ) -> (Vec<f32>, Option<Vec<Vec<f32>>>, Option<Vec<f32>>) {
        let new_error = input
            .iter()
            .zip(error.iter())
            .map(|(xi, yi)| if *xi > 0.0 { *yi } else { 0.0 })
            .collect();
        (new_error, None, None)
    }
}

struct Linear {
    dim_in: usize,
    dim_out: usize,
    weights: Vec<Vec<f32>>,
}

impl Linear {
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
                .map(|x| *x)
                .reduce(|acc, x| acc.iter().zip(x.iter()).map(|(i, j)| i + j).collect())
                .unwrap(),
            Some(weight_error),
            None,
        )
    }
}
