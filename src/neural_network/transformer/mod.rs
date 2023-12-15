use std::cmp;

use super::core::{add, linear_transform, matmul, Layer};

// ingredients for attention head
// query vector embedding matrix
// key vector embedding matrix
// matrix multiplication
// softmax
// value vector embedding matrix

fn attention_naive(q: Vec<f32>, k: Vec<f32>, v: Vec<f32>) -> Vec<f32> {
    let value = q
        .iter()
        .zip(k.iter())
        .map(|(x, y)| x * y)
        .fold(0.0, |acc, x| acc + x)
        / (k.len() as f32).sqrt();
    v.iter().map(|x| x * value).collect()
}

#[derive(Clone)]
pub struct Encoding {
    pub q: Vec<Vec<f32>>,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

fn self_attention_head_naive(sequence: Vec<Vec<f32>>, encoding: Encoding) -> Vec<Vec<f32>> {
    sequence
        .iter()
        .map(|x| {
            sequence
                .iter()
                .map(|y| {
                    attention_naive(
                        linear_transform(&encoding.q, &x, None),
                        linear_transform(&encoding.k, &y, None),
                        linear_transform(&encoding.v, &y, None),
                    )
                })
                .reduce(|acc, y| add(&acc, &y))
                .unwrap()
        })
        .collect()
}

fn head_dot(q: &[Vec<f32>], k: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let unnormalised = matmul(q, k, Some(1.0 / (q[0].len() as f32).sqrt()), true);
    let partition: f32 = unnormalised
        .iter()
        .map(|x| x.iter().map(|y| y.exp()).sum::<f32>())
        .sum();
    unnormalised
        .iter()
        .map(|x| x.iter().map(|y| y.exp() / partition).collect())
        .collect()
}

fn self_attention_head(sequence: &[Vec<f32>], encoding: &Encoding) -> Vec<Vec<f32>> {
    let q = matmul(&sequence, &encoding.q, None, true);
    let k = matmul(&sequence, &encoding.k, None, true);
    let v = matmul(&sequence, &encoding.v, None, true);
    let dot_prod = head_dot(&q, &k);
    matmul(&dot_prod, &v, None, false)
}

fn multi_attention(
    sequence: &[Vec<f32>],
    encoding: &[Encoding],
    projection: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let heads: Vec<Vec<Vec<f32>>> = encoding
        .iter()
        .map(|x| self_attention_head(sequence, x))
        .collect();
    let mut concat = Vec::new();
    for i in 0..sequence.len() {
        let row = heads
            .iter()
            .map(|x| x[i].clone())
            .collect::<Vec<Vec<f32>>>()
            .concat();
        concat.push(row);
    }
    matmul(&concat, projection, None, false)
}

fn multiply_forward(left: &[Vec<f32>], right: &[Vec<f32>]) -> Vec<Vec<f32>> {
    matmul(left, right, None, false)
}

fn multiply_back(
    left: &[Vec<f32>],
    right: &[Vec<f32>],
    error: &[Vec<f32>],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    // left: i * j, right: j*k, error: i*k
    let left_error = matmul(error, right, None, true);
    let left_t: Vec<Vec<f32>> = right
        .iter()
        .enumerate()
        .map(|(j, _)| left.iter().map(|x| x[j]).collect())
        .collect();
    let right_error = matmul(&left_t, error, None, false);
    let input_error = matmul(error, &multiply_forward(left, right), None, true);
    (input_error, left_error, right_error)
}
