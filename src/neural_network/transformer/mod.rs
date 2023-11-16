use std::cmp;

// ingredients for attention head
// query vector embedding matrix
// key vector embedding matrix
// matrix multiplication
// softmax
// value vector embedding matrix

#[test]
fn test_matmul_t() {
    let a = vec![vec![1.0, 2.0], vec![2.0, 3.0]];
    let b = vec![vec![2.0, 2.0], vec![1.0, 4.0]];
    let d = matmul(&a, &b, None, true);
    let expected = vec![vec![6.0, 9.0], vec![10.0, 14.0]];
    assert_eq!(d, expected);
}

fn matmul(a: &[Vec<f32>], b: &[Vec<f32>], scalar: Option<f32>, transpose: bool) -> Vec<Vec<f32>> {
    let mut result = Vec::new();
    let c = match scalar {
        Some(x) => x,
        None => 1.0,
    };
    for i in 0..a.len() {
        let mut row = Vec::new();
        for j in 0..b.len() {
            row.push(
                (0..a[0].len())
                    .collect::<Vec<_>>()
                    .iter()
                    .map(|k| {
                        if transpose {
                            c * a[i][*k] * b[j][*k]
                        } else {
                            c * a[i][*k] * b[*k][j]
                        }
                    })
                    .fold(0.0, |acc, x| acc + x),
            )
        }
        result.push(row)
    }
    result
}

fn attention_naive(q: Vec<f32>, k: Vec<f32>, v: Vec<f32>) -> Vec<f32> {
    let value = q
        .iter()
        .zip(k.iter())
        .map(|(x, y)| x * y)
        .fold(0.0, |acc, x| acc + x)
        / (k.len() as f32).sqrt();
    v.iter().map(|x| x * value).collect()
}

#[test]
fn test_linear_transform() {
    let a = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let x = vec![2.0, 3.0, 5.0];
    assert_eq!(linear_transform(&a, &x, None), x);
}

fn linear_transform(a: &[Vec<f32>], x: &[f32], b: Option<&[f32]>) -> Vec<f32> {
    a.iter()
        .map(|y| {
            y.iter()
                .zip(x.iter())
                .map(|(z, w)| z * w)
                .fold(0.0, |acc, y| acc + y)
        })
        .collect()
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
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
