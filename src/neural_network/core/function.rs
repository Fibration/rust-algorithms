use super::traits::{Activation, Derivative, Loss};

#[derive(Clone, Copy)]
pub enum Function {
    ReLU,
    CrossEntropy,
}

impl Activation for Function {
    fn activation(&self) -> fn(&[f64]) -> Vec<f64> {
        match self {
            Self::CrossEntropy => |x: &[f64]| {
                let denom = x.iter().map(|y| y.exp()).fold(0.0, |acc, y| acc + y);
                x.iter().map(|y| y.exp() / denom).collect()
            },
            Self::ReLU => |x| {
                x.iter()
                    .map(|xi| if *xi > 0.0 { *xi } else { 0.0 })
                    .collect()
            },
        }
    }
}

impl Derivative for Function {
    fn derivative(&self) -> fn(&[f64], &[f64]) -> Vec<f64> {
        match self {
            Self::CrossEntropy => |x, y| x.iter().zip(y.iter()).map(|(xi, yi)| *yi - *xi).collect(),
            Self::ReLU => |x, y| {
                x.iter()
                    .zip(y.iter())
                    .map(|(xi, yi)| if *xi > 0.0 { *yi } else { 0.0 })
                    .collect()
            },
        }
    }
}

impl Loss for Function {
    fn loss(&self) -> Option<fn(&[f64], &[f64]) -> f64> {
        match self {
            Self::CrossEntropy => Some(|x, y| {
                x.iter()
                    .zip(y.iter())
                    .map(|(xi, yi)| xi * yi.ln())
                    .fold(0.0, |acc, z| acc + z)
                    / (x.len() as f64)
            }),
            Self::ReLU => None,
        }
    }
}
