use crate::gen_range;
use crate::Value;
use crate::ValueVec;
use crate::MLP;
use std::collections::HashSet;
use std::f64::INFINITY;

fn sample_of_size<T, U>(inputs: &Vec<T>, targets: &Vec<U>, batch_size: usize) -> (Vec<T>, Vec<U>)
where
    T: Clone,
    U: Clone,
{
    assert!(!inputs.is_empty());
    assert!(inputs.len() == targets.len());
    assert!(batch_size <= inputs.len());
    let mut indices = HashSet::new();
    while indices.len() < batch_size {
        let i = gen_range(0, inputs.len());
        indices.insert(i);
    }
    let sample_indices = indices.into_iter().collect::<Vec<usize>>();
    let sample_inputs: Vec<T> = sample_indices.iter().map(|&i| inputs[i].clone()).collect();
    let sample_targets: Vec<U> = sample_indices.iter().map(|&i| targets[i].clone()).collect();
    (sample_inputs, sample_targets)
}

pub fn mean_squared_error(
    model: &MLP,
    inputs: Vec<ValueVec>,
    targets: Vec<ValueVec>,
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
) -> f64 {
    assert!(!inputs.is_empty());
    assert!(inputs.len() == targets.len());
    let mut last_loss = INFINITY;
    for _ in 0..epochs {
        let (sample_inputs, sample_targets) = sample_of_size(&inputs, &targets, batch_size);
        let predictions: Vec<ValueVec> = sample_inputs
            .iter()
            .map(|x| model.forward(x.clone()))
            .collect();
        let loss: Value = predictions
            .into_iter()
            .zip(&sample_targets)
            .map(|(prediction, target)| (prediction - target).pow(&ValueVec::from(vec![2.0])))
            .collect::<ValueVec>()
            .mean();
        last_loss = loss.data();
        println!("Loss: {}", last_loss);
        model.zero_grad();
        loss.backward();
        model.update(-learning_rate);
    }
    last_loss
}

pub fn binary_crossentropy(
    model: &MLP,
    inputs: Vec<ValueVec>,
    targets: Vec<bool>,
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
) -> f64 {
    assert!(!inputs.is_empty());
    assert!(inputs.len() == targets.len());
    let mut last_loss = INFINITY;
    for _ in 0..epochs {
        let (sample_inputs, sample_targets) = sample_of_size(&inputs, &targets, batch_size);
        let predictions: Vec<ValueVec> = sample_inputs
            .iter()
            .map(|x| model.forward(x.clone()))
            .collect();
        let targets_as_values: Vec<Value> =
            sample_targets.iter().map(|&x| Value::from(x)).collect();
        let loss: Value = predictions
            .into_iter()
            .zip(&targets_as_values)
            .map(|(prediction, target)| {
                let prob = prediction.sigmoid();
                let loss = target * prob.log()
                    + (Value::from(1.0) - target) * (Value::from(1.0) - prob).log();
                -loss
            })
            .collect::<ValueVec>()
            .mean();
        last_loss = loss.data();
        model.zero_grad();
        loss.backward();
        model.update(-learning_rate);
    }
    last_loss
}

pub fn categorical_crossentropy(
    model: &MLP,
    inputs: Vec<ValueVec>,
    targets: Vec<usize>,
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
) -> f64 {
    assert!(!inputs.is_empty());
    assert!(inputs.len() == targets.len());
    let mut last_loss = INFINITY;
    for _ in 0..epochs {
        let (sample_inputs, sample_targets) = sample_of_size(&inputs, &targets, batch_size);
        let predictions: Vec<ValueVec> = sample_inputs
            .iter()
            .map(|x| model.forward(x.clone()))
            .collect();
        let loss: Value = predictions
            .into_iter()
            .zip(&sample_targets)
            .map(|(prediction, target)| {
                let nll = -prediction.softmax().log();
                assert!(*target < nll.len());
                nll[*target].clone()
            })
            .collect::<ValueVec>()
            .mean();
        last_loss = loss.data();
        model.zero_grad();
        loss.backward();
        model.update(-learning_rate);
    }
    last_loss
}
