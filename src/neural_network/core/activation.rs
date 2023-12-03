use super::Layer;

pub struct ReLU {
    dim_in: usize,
    dim_out: usize,
}

impl Layer for ReLU {
    fn dim_in(&self) -> usize {
        self.dim_in
    }
    fn dim_out(&self) -> usize {
        self.dim_out
    }

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
