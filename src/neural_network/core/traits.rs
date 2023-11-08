use super::Function;

pub trait Activation {
    fn activation(&self) -> fn(&[f64]) -> Vec<f64>;
}

pub trait Derivative {
    fn derivative(&self) -> fn(&[f64], &[f64]) -> Vec<f64>;
}

pub trait Loss {
    fn loss(&self) -> Option<fn(&[f64], &[f64]) -> f64>;
}

pub trait Layer: Clone {
    fn cap(&self) -> Function;
    fn forward(&self, input: &[f64]) -> Vec<f64>;
    fn back(&self, output: &[f64], error: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>);
}
