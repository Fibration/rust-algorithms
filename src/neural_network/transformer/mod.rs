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
    let d = matmul(a, b, None, true);
    let expected = vec![vec![6.0, 9.0], vec![10.0, 14.0]];
    assert_eq!(d, expected);
}

fn matmul(
    a: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
    scalar: Option<f32>,
    transpose: bool,
) -> Vec<Vec<f32>> {
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
    assert_eq!(linear_transform(&a, &x), x);
}

fn linear_transform(a: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    a.iter()
        .map(|y| {
            y.iter()
                .zip(x.iter())
                .map(|(z, w)| z * w)
                .fold(0.0, |acc, y| acc + y)
        })
        .collect()
}
