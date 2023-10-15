use std::collections::HashMap;

use rand_distr::num_traits::Float;

#[test]
fn test_entropy() {
    let mut sample = Vec::<f32>::new();
    let mut labels = Vec::<u8>::new();
    for i in 0..100 {
        sample.push(i as f32);
        if i % 3 == 0 {
            labels.push(0);
        } else {
            labels.push(1);
        }
    }
    assert_eq!(entropy(sample, labels), 0.9248187);
}

fn entropy(_sample: Vec<f32>, labels: Vec<u8>) -> f32 {
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
