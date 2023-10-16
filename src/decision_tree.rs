use std::collections::HashMap;

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
