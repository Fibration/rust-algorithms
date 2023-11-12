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
