use super::core::{Layer, Loss};

fn SGD(
    network: Vec<impl Layer>,
    data: &[Vec<f32>],
    labels: &[Vec<f32>],
) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
    // run forwards
    let mut forwards = Vec::new();
    for j in 0..network.len() {
        let layer_results: Vec<Vec<f32>> = (0..data.len())
            .collect::<Vec<_>>()
            .iter()
            .map(|i| network[j].forward(&data[*i]))
            .collect();
        forwards.push(layer_results);
    }

    // calculate loss
    let mut _loss = 0.0;
    for i in 0..labels.len() {
        _loss += match network[network.len()].cap().loss() {
            Some(f) => f(&forwards[forwards.len()][i], &labels[i]),
            None => 0.0,
        };
    }

    // run back
    let mut backwards = Vec::new();
    let mut bias = Vec::new();
    let mut error = Vec::from(labels);
    for i in (0..network.len()).rev() {
        let result: Vec<(Vec<Vec<f32>>, Vec<f32>, Vec<f32>)> = (0..labels.len())
            .collect::<Vec<_>>()
            .iter()
            .map(|j| network[i].back(&forwards[i][*j], &error[*j]))
            .collect();
        error = result.iter().map(|x| x.2.clone()).collect();

        backwards.push(
            result
                .iter()
                .map(|x| x.0.clone())
                .fold(Vec::new(), |acc, x| add(x, acc)),
        );
        bias.push(
            result
                .iter()
                .map(|x| x.1.clone())
                .fold(Vec::new(), |acc, x| vector_add(x, acc)),
        );
    }

    (backwards, bias)
}

fn add(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut c = Vec::new();
    // HACKY
    if b.len() == 0 {
        return a;
    }
    for i in 0..a.len() {
        let mut row = Vec::new();
        for j in 0..a[0].len() {
            row.push(a[i][j] + b[i][j]);
        }
        c.push(row);
    }
    c
}

fn vector_add(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    let mut c = Vec::new();
    // HACKY
    if b.len() == 0 {
        return a;
    }
    for i in 0..a.len() {
        c.push(a[i] + b[i]);
    }
    c
}

fn divide(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, scalar: Option<f32>) -> Vec<Vec<f32>> {
    let mut c = Vec::new();
    // HACKY
    if b.len() == 0 {
        return a;
    }
    let _scalar = match scalar {
        Some(x) => x,
        None => 1.0,
    };
    for i in 0..a.len() {
        let mut row = Vec::new();
        for j in 0..a[0].len() {
            row.push(_scalar * a[i][j] / b[i][j]);
        }
        c.push(row);
    }
    c
}

fn rmsprop(
    network: Vec<impl Layer>,
    data: &[Vec<f32>],
    labels: &[Vec<f32>],
    averages: (&[Vec<Vec<f32>>], &[Vec<f32>]),
    learning_rate: f32,
) -> (
    Vec<Vec<Vec<f32>>>,
    Vec<Vec<f32>>,
    Vec<Vec<Vec<f32>>>,
    Vec<Vec<f32>>,
) {
    let sgd_results = SGD(network, data, labels);
    let square_grad: Vec<Vec<Vec<f32>>> = sgd_results
        .0
        .iter()
        .map(|x| {
            x.iter()
                .map(|y| y.iter().map(|z| 0.1 * z * z).collect())
                .collect()
        })
        .collect();
    let square_bias: Vec<Vec<f32>> = sgd_results
        .1
        .iter()
        .map(|x| x.iter().map(|y| 0.1 * y * y).collect())
        .collect();

    let new_grad_average: Vec<Vec<Vec<f32>>> = averages
        .0
        .iter()
        .enumerate()
        .map(|(i, x)| {
            add(
                x.clone()
                    .iter()
                    .map(|y| y.iter().map(|z| 0.9 * z).collect())
                    .collect(),
                square_grad[i].clone(),
            )
        })
        .collect();
    let new_bias_average = add(
        averages
            .1
            .to_vec()
            .iter()
            .map(|y| y.iter().map(|z| 0.9 * z).collect())
            .collect(),
        square_bias,
    );
    let grad = sgd_results
        .0
        .iter()
        .enumerate()
        .map(|(i, x)| divide(x.to_vec(), new_grad_average[i].clone(), Some(learning_rate)))
        .collect();
    let bias = divide(sgd_results.1, new_bias_average.clone(), Some(learning_rate));

    (grad, bias, new_grad_average, new_bias_average)
}
