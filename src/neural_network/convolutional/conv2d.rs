use crate::neural_network::core::Layer2D;

use super::{convolution, matrix_op, matrix_rotate, pad_around, pad_right_within};

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
                    .reduce(|acc, x| matrix_op(&acc, &x, |y, z| y + z))
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
        let adjusted_error: Vec<Vec<Vec<f32>>> = error
            .iter()
            .map(|x| pad_right_within(&x, (self.stride.0 - 1, self.stride.1 - 1)))
            .collect();

        let error_by_input = input
            .iter()
            .map(|i| {
                self.filters
                    .iter()
                    // get contribution of each input to the error by
                    // calculating the convolution of the input by the filter
                    // then taking the entrywise product with the relevant error;
                    // this is then convolved with the filter to get the input_error
                    // then summed entrywise for that input
                    .zip(adjusted_error.iter())
                    .map(|(filter, filter_error)| {
                        matrix_op(
                            &convolution(i, filter, self.padding, self.stride),
                            filter_error,
                            |x, y| x * y,
                        )
                    })
                    .zip(self.filters.iter())
                    .map(|(error_channel, filter)| {
                        convolution(
                            &pad_around(
                                error_channel,
                                (filter.len() - 1, filter[0].len() - 1),
                                (self.stride.0 - 1, self.stride.1 - 1),
                            ),
                            &matrix_rotate(filter),
                            (0, 0),
                            (1, 1),
                        )
                    })
                    .reduce(|acc, err| matrix_op(&acc, &err, |x, y| x + y))
                    .unwrap()
            })
            .collect();

        let filter_error = self
            .filters
            .iter()
            .zip(adjusted_error.iter())
            .map(|(filter, filter_error)| {
                input
                    .iter()
                    .map(|input_piece| {
                        convolution(
                            input_piece,
                            &matrix_op(
                                &convolution(input_piece, filter, self.padding, self.stride),
                                &filter_error,
                                |x, y| x * y,
                            ),
                            self.padding,
                            (1, 1),
                        )
                    })
                    .reduce(|acc, x| matrix_op(&acc, &x, |y, z| y + z))
                    .unwrap()
            })
            .collect();

        (error_by_input, Some(filter_error), None)
    }
}
