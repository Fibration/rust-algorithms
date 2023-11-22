#[cfg(test)]
mod test_core {
    use crate::neural_network::core::array::{stack, unstack};

    #[test]
    fn test_stack() {
        assert_eq!(
            stack(&[1.0, 1.0, 1.0, 1.0], (2, 2)),
            Vec::from([[1.0, 1.0], [1.0, 1.0]])
        );
    }

    #[test]
    fn test_unstack() {
        let matrix: Vec<Vec<f32>> = Vec::from([vec![1.0, 1.0], vec![1.0, 1.0]]);
        assert_eq!(unstack(&matrix), [1.0, 1.0, 1.0, 1.0]);
    }
}
