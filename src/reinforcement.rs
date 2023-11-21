use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn test_bandit() {
    let mut history = vec![0.0, 0.0, 0.0, 0.0];
    let mut result: usize;
    (result, history) = bandit((0, 0.0), &history, 0.1, None);
    assert_eq!(result, 0);
    (result, _) = bandit((0, -0.5), &history, 0.1, None);
    assert_ne!(result, 0);
}

fn bandit(
    result: (usize, f32),
    history: &[f32],
    alpha: f32,
    seed: Option<ChaCha8Rng>,
) -> (usize, Vec<f32>) {
    let book: Vec<f32> = history
        .iter()
        .enumerate()
        .map(|(i, x)| {
            if i == result.0 {
                0.9 * x + 0.1 * result.1
            } else {
                *x
            }
        })
        .collect();

    let mut rng = match seed {
        Some(x) => x,
        None => ChaCha8Rng::seed_from_u64(1),
    };
    if rng.gen_range(0.0..1.0) < alpha {
        (rng.gen_range(0..book.len()), book)
    } else {
        let best = book.iter().map(|x| *x).reduce(f32::max).unwrap();
        (book.iter().position(|x| *x == best).unwrap(), book)
    }
}
