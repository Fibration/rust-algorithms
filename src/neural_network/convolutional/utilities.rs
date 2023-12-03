pub fn convolution(
    data: &[Vec<f32>],
    filter: &[Vec<f32>],
    padding: (usize, usize),
    stride: (usize, usize),
) -> Vec<Vec<f32>> {
    let mut padded = Vec::new();
    for _ in 0..padding.0 {
        let mut row = Vec::new();
        for _ in 0..(data[0].len() + 2 * padding.1) {
            row.push(0.0);
        }
        padded.push(row);
    }
    for i in 0..data.len() {
        let mut row = Vec::new();
        for _ in 0..padding.1 {
            row.push(0.0);
        }
        for j in 0..data[i].len() {
            row.push(data[i][j]);
        }
        for _ in 0..padding.1 {
            row.push(0.0);
        }
        padded.push(row);
    }
    for _ in 0..padding.0 {
        let mut row = Vec::new();
        for _ in 0..(data[0].len() + 2 * padding.1) {
            row.push(0.0);
        }
        padded.push(row);
    }
    let mut output = Vec::new();
    for i in 0..((padded.len() - filter.len() + stride.0) / stride.0) {
        let mut row = Vec::new();
        for j in 0..(padded[0].len() - filter[0].len() + stride.1) / stride.1 {
            let mut convolution = 0.0;
            for k in 0..filter.len() {
                for l in 0..filter[0].len() {
                    convolution += filter[k][l] * padded[k + i * stride.0][l + j * stride.1];
                }
            }
            row.push(convolution);
        }
        output.push(row);
    }
    output
}

pub fn pad_right_within(matrix: &[Vec<f32>], padding: (usize, usize)) -> Vec<Vec<f32>> {
    let new_row_len = matrix[0].len() * (1 + padding.1);
    let mut empty = Vec::new();
    for _ in 0..(new_row_len) {
        empty.push(0.0);
    }
    let mut padded = Vec::new();
    for i in 0..matrix.len() {
        let mut padded_row = Vec::new();
        for j in 0..matrix[0].len() {
            padded_row.push(matrix[i][j]);
            for _ in 0..padding.1 {
                padded_row.push(0.0);
            }
        }
        padded.push(padded_row);
        for _ in 0..padding.0 {
            padded.push(empty.clone());
        }
    }
    padded
}

pub fn pad_around(
    matrix: Vec<Vec<f32>>,
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Vec<Vec<f32>> {
    let new_row_len = padding.1 * 2 + matrix[0].len() + (matrix[0].len() - 1) * dilation.1;
    let mut empty = Vec::new();
    for _ in 0..(new_row_len) {
        empty.push(0.0);
    }
    let mut padded = Vec::new();
    for _ in 0..padding.0 {
        padded.push(empty.clone());
    }
    for i in 0..matrix.len() {
        let mut padded_row = Vec::new();
        // pad the left
        for _ in 0..padding.1 {
            padded_row.push(0.0);
        }
        for j in 0..(matrix[0].len() - 1) {
            // add column and then dilation
            padded_row.push(matrix[i][j]);
            for _ in 0..dilation.1 {
                padded_row.push(0.0);
            }
        }
        // add final column entry
        padded_row.push(matrix[i][matrix[0].len() - 1]);
        // pad the right
        for _ in 0..padding.1 {
            padded_row.push(0.0);
        }
        padded.push(padded_row);
        // dilate next row unless last row
        if i < matrix.len() - 1 {
            for _ in 0..dilation.0 {
                padded.push(empty.clone());
            }
        }
    }

    for _ in 0..padding.0 {
        padded.push(empty.clone());
    }

    padded
}

pub fn matrix_op(a: &[Vec<f32>], b: &[Vec<f32>], func: fn(&f32, &f32) -> f32) -> Vec<Vec<f32>> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.iter().zip(y.iter()).map(|(j, k)| func(j, k)).collect())
        .collect()
}



pub fn matrix_rotate(x: &[Vec<f32>]) -> Vec<Vec<f32>> {
    x.iter()
        .map(|y| y.iter().rev().map(|z| *z).collect())
        .rev()
        .collect()
}
