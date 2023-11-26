use crate::neural_network::core::Layer2D;

use super::convolution;

#[derive(Clone)]
struct Conv2D {
    dim_in: (usize, usize, usize),
    dim_out: (usize, usize, usize),
    filters: Vec<Vec<Vec<f32>>>,
    padding: (usize, usize),
    stride: (usize, usize),
}

impl Layer2D for Conv2D {
    fn dim_in(&self) -> (usize, usize, usize) {
        self.dim_in
    }
    fn dim_out(&self) -> (usize, usize, usize) {
        self.dim_out
    }
    fn forward(&self, input: &[Vec<Vec<f32>>]) -> Vec<Vec<Vec<f32>>> {
        self.filters
            .iter()
            .map(|filter| {
                input
                    .iter()
                    .map(|piece| convolution(piece, filter, self.padding, self.stride))
                    .reduce(|acc, x| matrix_sum(&acc, &x))
                    .unwrap()
            })
            .collect()
    }
    fn back(
        &self,
        input: &[Vec<Vec<f32>>],
        error: &[Vec<Vec<f32>>],
    ) -> (
        Vec<Vec<Vec<f32>>>,
        Option<Vec<Vec<Vec<f32>>>>,
        Option<Vec<f32>>,
    ) {
        todo!()
    }
}

fn matrix_sum(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.iter().zip(y.iter()).map(|(j, k)| j + k).collect())
        .collect()
}
