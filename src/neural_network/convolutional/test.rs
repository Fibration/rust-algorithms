#[cfg(test)]
mod test_conv {
    use crate::neural_network::{
        convolutional::{
            convolution, matrix_rotate, pad_around, pad_right_within, ConvolutionLayer,
        },
        core::{stack, Function, Layer},
    };

    #[test]
    fn test_pad_right_within() {
        let mut matrix = Vec::new();
        matrix.push(vec![1.0, 1.0]);
        matrix.push(vec![1.0, 1.0]);
        let mut expected = Vec::new();
        expected.push(vec![1.0, 0.0, 1.0, 0.0]);
        expected.push(vec![0.0, 0.0, 0.0, 0.0]);
        expected.push(vec![1.0, 0.0, 1.0, 0.0]);
        expected.push(vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(pad_right_within(&matrix, (1, 1)), expected);
    }

    #[test]
    fn test_pad_around() {
        let mut matrix = Vec::new();
        matrix.push(vec![1.0, 1.0]);
        matrix.push(vec![1.0, 1.0]);
        let mut expected = Vec::new();
        expected.push(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        expected.push(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        expected.push(vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        expected.push(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        expected.push(vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        expected.push(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        expected.push(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(pad_around(matrix, (2, 2), (1, 1)), expected);
    }

    #[test]
    fn test_convlayer_forward() {
        let input = vec![-2.0, -3.0, 2.0, 1.0, 1.5, 2.5, 2.5, 1.5, 1.0, 2.0, 2.0, 1.0];
        let mut conv =
            ConvolutionLayer::new((3, 4), (2, 3), (2, 2), (0, 0), (1, 1), Function::ReLU);
        conv.a = Vec::new();
        conv.a.push(vec![1.0, 1.0]);
        conv.a.push(vec![1.0, 1.0]);
        let result = conv.forward(&input);
        assert_eq!(result, [0.0, 4.0, 7.0, 7.0, 9.0, 7.0]);
    }

    #[test]
    fn test_convlayer_back() {
        let mut conv =
            ConvolutionLayer::new((3, 4), (2, 3), (2, 2), (0, 0), (1, 1), Function::ReLU);
        conv.a = Vec::new();
        conv.a.push(vec![1.0, 1.0]);
        conv.a.push(vec![1.0, 1.0]);
        let input = vec![-2.0, -3.0, 2.0, 1.0, 1.5, 2.5, 2.5, 1.5, 1.0, 2.0, 2.0, 1.0];
        let error = vec![0.0, 1.0, 2.0, 1.0, 1.0, 1.0];
        let back = conv.back(&input, &error);
        let expected_partial = stack(&[7.5, 10.5, 12.5, 10.5], (2, 2));
        let expected_error = [0.0, 1.0, 3.0, 2.0, 1.0, 3.0, 5.0, 3.0, 1.0, 2.0, 2.0, 1.0];
        assert_eq!(back.0, expected_partial);
        assert_eq!(back.1, [1.0]);
        assert_eq!(back.2, expected_error);
    }

    #[test]
    fn test_convolution() {
        let mut input = Vec::new();
        input.push(vec![1.0, 2.0, 2.0, 1.0]);
        input.push(vec![1.5, 2.5, 2.5, 1.5]);
        input.push(vec![1.5, 2.5, 2.5, 1.5]);
        input.push(vec![1.0, 2.0, 2.0, 1.0]);
        let mut filter = Vec::new();
        filter.push(vec![1.0, 1.0]);
        filter.push(vec![1.0, 1.0]);
        let output = convolution(&input, &filter, (1, 1), (2, 2));
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].len(), 3);
        assert_eq!(output[0], [1.0, 4.0, 1.0]);
        assert_eq!(output[1], [3.0, 10.0, 3.0]);
        assert_eq!(output[2], [1.0, 4.0, 1.0]);
    }

    #[test]
    fn test_matrix_rotate() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let x_rotated = vec![vec![4.0, 3.0], vec![2.0, 1.0]];
        assert_eq!(matrix_rotate(&x), x_rotated)
    }
}
