use rand::thread_rng;
use rand_distr::{Distribution, Normal};

pub fn he_initialise(dim_in: usize, dim_out: usize) -> Vec<f32> {
    let factor = (2.0 / (dim_in as f32)).sqrt();
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, factor).unwrap();

    (0..dim_out)
        .collect::<Vec<_>>()
        .iter()
        .map(|_| normal.sample(&mut rng))
        .collect()
}

#[test]
fn test_matmul_t() {
    let a = vec![vec![1.0, 2.0], vec![2.0, 3.0]];
    let b = vec![vec![2.0, 2.0], vec![1.0, 4.0]];
    let d = matmul(&a, &b, None, true);
    let expected = vec![vec![6.0, 9.0], vec![10.0, 14.0]];
    assert_eq!(d, expected);
}

pub fn matmul(
    a: &[Vec<f32>],
    b: &[Vec<f32>],
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

pub fn linear_transform(a: &[Vec<f32>], x: &[f32], b: Option<&[f32]>) -> Vec<f32> {
    a.iter()
        .map(|y| {
            y.iter()
                .zip(x.iter())
                .map(|(z, w)| z * w)
                .fold(0.0, |acc, y| acc + y)
        })
        .collect()
}

pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub fn outer(a: &[f32], b: &[f32]) -> Vec<Vec<f32>> {
    a.iter()
        .map(|x| b.iter().map(|y| x * y).collect())
        .collect()
}
