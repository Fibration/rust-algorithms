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
