use super::core::{Layer, Loss};

fn SGD(
    network: Vec<impl Layer>,
    data: &[Vec<f64>],
    labels: &[Vec<f64>],
    learning_rate: f64,
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut updated = network.clone();

    // run forwards
    let mut forwards = Vec::new();
    for layer in network {
        let layer_results: Vec<Vec<f64>> = (0..data.len())
            .collect::<Vec<_>>()
            .iter()
            .map(|i| layer.forward(&data[*i]))
            .collect();
        forwards.push(layer_results);
    }

    // calculate loss
    let mut loss = 0.0;
    for i in 0..labels.len() {
        loss += match network[network.len()].cap().loss() {
            Some(f) => f(&forwards[forwards.len()][i], &labels[i]),
            None => 0.0,
        };
    }

    // run back
    let mut backwards = Vec::new();
    let mut bias = Vec::new();
    let mut error = Vec::from(labels);
    for i in (0..network.len()).rev() {
        let result: Vec<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>)> = (0..labels.len())
            .collect::<Vec<_>>()
            .iter()
            .map(|j| network[i].back(&forwards[i][*j], &error[*j]))
            .collect();
        error = result.iter().map(|x| x.2).collect();

        backwards.push(
            result
                .iter()
                .map(|x| x.0)
                .fold(Vec::new(), |acc, x| add(x, acc)),
        );
        bias.push(
            result
                .iter()
                .map(|x| x.1)
                .fold(Vec::new(), |acc, x| vector_add(x, acc)),
        );
    }

    (backwards, bias)
}

fn add(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

fn vector_add(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
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
