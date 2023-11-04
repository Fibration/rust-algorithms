
pub fn convolution(
    data: &[Vec<f64>],
    filter: &[Vec<f64>],
    padding: (usize, usize),
    stride: (usize, usize),
) -> Vec<Vec<f64>> {
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
