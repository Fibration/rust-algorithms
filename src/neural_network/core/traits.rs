use super::Function;

pub trait Activation {
    fn activation(&self) -> fn(&[f32]) -> Vec<f32>;
}

pub trait Derivative {
    fn derivative(&self) -> fn(&[f32], &[f32]) -> Vec<f32>;
}

pub trait Loss {
    fn loss(&self) -> Option<fn(&[f32], &[f32]) -> f32>;
}

pub trait Layer: Clone {
    fn cap(&self) -> Function;
    fn forward(&self, input: &[f32]) -> Vec<f32>;
    fn back(&self, output: &[f32], error: &[f32]) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>);
}

pub trait Layer2D: Clone {
    fn dim_in(&self) -> (usize, usize, usize);
    fn dim_out(&self) -> (usize, usize, usize);
    fn forward(&self, input: &[Vec<Vec<f32>>]) -> Vec<Vec<Vec<f32>>>;
    fn back(
        &self,
        input: &[Vec<Vec<f32>>],
        error: &[Vec<Vec<f32>>],
    ) -> (
        Vec<Vec<Vec<f32>>>,
        Option<Vec<Vec<Vec<f32>>>>,
        Option<Vec<f32>>,
    );
}
