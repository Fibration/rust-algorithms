use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};

#[test]
fn test_svm() {
    let (points, labels) = sample(
        50,
        50,
        vec![(1.0, 1.0), (5.0, 1.0)],
        vec![(5.0, 1.0), (1.0, 1.0)],
    );
    let model = train_svm(points.clone(), labels);
    assert_eq!(svm_predict(model.0, model.1, points[0].clone()), -1.0);
}

fn sample(
    size1: u32,
    size2: u32,
    params1: Vec<(f64, f64)>,
    params2: Vec<(f64, f64)>,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();
    let mut rng = thread_rng();
    for i in 0..size1 {
        data.push(
            params1
                .iter()
                .map(|(m, v)| Normal::new(*m, *v).unwrap().sample(&mut rng))
                .collect(),
        );
        labels.push(-1.0);
    }
    for i in 0..size2 {
        data.push(
            params2
                .iter()
                .map(|(m, v)| Normal::new(*m, *v).unwrap().sample(&mut rng))
                .collect(),
        );
        labels.push(1.0);
    }
    return (data, labels);
}

fn train_svm(data: Vec<Vec<f64>>, labels: Vec<f64>) -> (Vec<f64>, f64) {
    let learning_rate = 0.01;
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let uniform: Uniform<u32> = Uniform::new(0, data.len() as u32);
    let mut w: Vec<f64> = Vec::new();
    let mut b = normal.sample(&mut rng);
    for _i in 0..data[0].len() as u32 {
        w.push(normal.sample(&mut rng));
    }
    for _ in 0..(data.len() / 16) {
        let mut sample_batch: Vec<usize> = Vec::new();
        for _j in 0..16 {
            sample_batch.push(uniform.sample(&mut rng) as usize);
        }
        print!("{sample_batch:?}");
        print!("{w:?}");
        let mut dw: Vec<f64> = Vec::new();
        let mut db: f64 = 0.0;
        for k in sample_batch {
            for i in 0..w.len() {
                dw.push(labels[k] * data[k][i]);
                w[i] = w[i] - learning_rate * w[i] + learning_rate * dw[i]
            }
            db += labels[k];
            b = b + learning_rate * db;
        }
    }
    return (w, b);
}

fn svm_predict(w: Vec<f64>, b: f64, point: Vec<f64>) -> f64 {
    let mut sum = b;
    for i in 0..w.len() {
        sum += w[i] * point[i];
    }
    return sum.signum();
}
