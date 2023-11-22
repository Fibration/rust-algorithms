use crate::neural_network::{
    core::Layer,
    linear::linear::{linear_nn, NeuralNetworkLayer},
};

#[cfg(test)]
mod test_linear {
    use crate::neural_network::core::Function;

    use super::*;

    #[test]
    fn test_neural_network_layer_forward() {
        let mut nn = NeuralNetworkLayer::new(5, 3, Function::ReLU);
        println!("{nn:?}");
        nn.a = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            vec![-1.0, -1.0, -1.0, -1.0, -1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        nn.b = vec![1.0, 1.0, 1.0];
        let result = nn.forward(&[5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(result, vec![36.0, 0.0, 36.0]);
    }

    #[test]
    fn test_neural_network_layer_back() {
        let mut nn = NeuralNetworkLayer::new(5, 3, Function::ReLU);
        println!("{nn:?}");
        nn.a = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            vec![-1.0, -1.0, -1.0, -1.0, -1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        nn.b = vec![1.0, 1.0, 1.0];
        let result = nn.forward(&[5.0, 6.0, 7.0, 8.0, 9.0]);
        let back_result = nn.back(&result[..], &[1.0, 1.0, 1.0]);
        assert_eq!(back_result.0.len() as u32, nn.dim_out);
        assert_eq!(back_result.0[0].len() as u32, nn.dim_in);
        assert_eq!(back_result.1.len() as u32, nn.dim_out);
        assert_eq!(back_result.2.len() as u32, nn.dim_in);
    }

    #[test]
    fn test_linear_nn() {
        let linear = linear_nn(&[16, 8, 4, 2], Function::ReLU, Function::CrossEntropy);
        let input: Vec<f32> = (0..16)
            .collect::<Vec<_>>()
            .iter()
            .map(|x| *x as f32)
            .collect();
        let result = linear[2].forward(&linear[1].forward(&linear[0].forward(&input[..])[..])[..]);
        println!("{result:?}");
        assert_eq!(linear.len(), 3);
        assert_eq!(linear[0].dim_in, 16);
        assert_eq!(linear[2].dim_out, 2);
        assert_eq!(linear[0].dim_out, linear[1].dim_in);
        assert_eq!(linear[1].dim_out, linear[2].dim_in);
    }
}
