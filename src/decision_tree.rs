use std::collections::HashMap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, Uniform};

#[test]
fn test_entropy() {
    let mut labels = Vec::<u8>::new();
    for i in 0..99 {
        if i % 3 == 0 {
            labels.push(0);
        } else {
            labels.push(1);
        }
    }
    assert_eq!(
        entropy(labels),
        -(1.0 / 3.0) * f32::log2(1.0 / 3.0) - (2.0 / 3.0) * f32::log2(2.0 / 3.0)
    );
}

fn entropy(labels: Vec<u8>) -> f32 {
    let total = labels.len() as f32;
    let mut unique_labels = HashMap::<u8, u32>::new();
    let mut ent = 0.0;
    for x in labels {
        unique_labels.insert(x, *(unique_labels.get(&x).unwrap_or_else(|| &0)) + 1);
    }
    for key in unique_labels.keys() {
        let prob = *unique_labels.get(key).unwrap() as f32 / total;
        ent -= prob * f32::log2(prob);
    }
    ent
}

#[test]
fn test_information_gain() {
    let mut sample = Vec::<String>::new();
    let mut labels = Vec::<u8>::new();
    for i in 0..99 {
        if i % 3 == 0 {
            sample.push(String::from("multiple of 3"));
            labels.push(0);
        } else {
            sample.push(String::from("non multiple of 3"));
            labels.push(1);
        }
    }
    let new_sample_labels = vec![
        labels
            .clone()
            .iter()
            .map(|x| *x)
            .filter(|x| *x == 0)
            .collect(),
        labels
            .clone()
            .iter()
            .map(|x| *x)
            .filter(|x| *x == 1)
            .collect(),
    ];

    assert_eq!(
        information_gain(labels, new_sample_labels),
        -(1.0 / 3.0) * f32::log2(1.0 / 3.0) - (2.0 / 3.0) * f32::log2(2.0 / 3.0)
    );
}

fn information_gain(pop1: Vec<u8>, pop2: Vec<Vec<u8>>) -> f32 {
    let mut new_entropies = 0.0;
    for x in pop2 {
        new_entropies += (x.len() as f32 / pop1.len() as f32) * entropy(x)
    }
    entropy(pop1) - new_entropies
}

#[test]
fn test_id3() {
    let tree = DecisionTree::from(vec![
        (0, (0, 0.5, 1, 2, None)),
        (1, (1, 0.5, 3, 4, None)),
        (2, (2, 0.5, 5, 6, None)),
        (3, (1, 0.5, 3, 4, Some(0.2))),
        (4, (1, 0.5, 3, 4, Some(0.4))),
        (5, (2, 0.5, 5, 6, Some(0.6))),
        (6, (2, 0.5, 5, 6, Some(0.8))),
    ]);
    let sample = vec![0.4, 0.6];
    assert_eq!(id3(sample, tree), 0.4);
}

fn id3(sample: Vec<f32>, tree: DecisionTree) -> f32 {
    let mut current_node: u32 = 0;
    loop {
        let split = tree.tree.get(&current_node).unwrap();
        if let Some(z) = split.4 {
            return z;
        } else {
            if sample[split.0 as usize] <= split.1 {
                current_node = split.2;
            } else {
                current_node = split.3;
            }
        }
    }
}

#[test]
fn test_id3_train() {
    let mut data = Vec::<Vec<f32>>::new();
    let mut labels = Vec::<f32>::new();
    let mut seed = ChaCha8Rng::seed_from_u64(22);
    let uniform = Uniform::from(0.0..1.0);
    let normal = Normal::new(0.0, 0.05).unwrap();

    for _ in 0..100 {
        let x = vec![
            uniform.sample(&mut seed),
            uniform.sample(&mut seed),
            uniform.sample(&mut seed),
        ];
        let delta_y = normal.sample(&mut seed);

        if x[0] <= 0.5 {
            if x[1] <= 0.5 {
                labels.push(0.2 + delta_y);
            } else {
                labels.push(0.4 + delta_y);
            }
        } else {
            if x[2] <= 0.5 {
                labels.push(0.6 + delta_y);
            } else {
                labels.push(0.8 + delta_y);
            }
        }
        data.push(x);
    }

    let tree = id3_train(data);
    assert!(id3(vec![0.1, 0.1, 0.1], tree.tree) < 0.3);
}

fn id3_train(data: Vec<Vec<f32>>) -> DecisionTree {
    DecisionTree::from(vec![
        (0, (0, 0.5, 1, 2, None)),
        (1, (1, 0.5, 3, 4, None)),
        (2, (2, 0.5, 5, 6, None)),
        (3, (1, 0.5, 3, 4, Some(0.2))),
        (4, (1, 0.5, 3, 4, Some(0.4))),
        (5, (2, 0.5, 5, 6, Some(0.6))),
        (6, (2, 0.5, 5, 6, Some(0.8))),
    ])
}

struct DecisionTree {
    tree: HashMap<u32, (u8, f32, u32, u32, Option<f32>)>,
}

impl DecisionTree {
    fn from(data: Vec<(u32, (u8, f32, u32, u32, Option<f32>))>) -> DecisionTree {
        DecisionTree { tree: data }
    }
    fn get_split_fields(&self) -> Vec<u8> {
        self.tree.iter().map(|x| x.1 .0).collect()
    }
}
