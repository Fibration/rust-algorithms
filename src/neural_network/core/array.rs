pub fn stack(flat: &[f32], dim: (usize, usize)) -> Vec<Vec<f32>> {
    let mut stacked = Vec::new();
    for i in 0..dim.0 {
        let mut row = Vec::new();
        for j in 0..dim.1 {
            row.push(flat[i * dim.1 + j]);
        }
        stacked.push(row);
    }
    stacked
}

pub fn unstack(matrix: &[Vec<f32>]) -> Vec<f32> {
    let mut flat = Vec::new();
    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            flat.push(matrix[i][j]);
        }
    }
    flat
}
