use microgradr::{
    mean_squared_error,
    sgd::{binary_crossentropy, categorical_crossentropy},
    v1d, Neuron, Value, MLP,
};

#[test]
fn test_neuron() {
    let neuron = Neuron::new(4, false);
    let parameters = neuron.parameters();
    assert_eq!(parameters.len(), 4 + 1);

    let mut neuron = Neuron::new(4, false);
    neuron.set_weights(&v1d![1.0, 2.0, 3.0, 4.0]);
    neuron.set_bias(&Value::from(5.0));
    let inputs = v1d![1.0, 2.0, 3.0, 4.0];
    let result = neuron.forward(&inputs);
    assert_eq!(result.data(), 35.0);

    let mut neuron = Neuron::without_bias(4, false);
    neuron.set_weights(&v1d![1.0, 2.0, 3.0, 4.0]);
    let inputs = v1d![1.0, 2.0, 3.0, 4.0];
    let result = neuron.forward(&inputs);
    assert_eq!(result.data(), 30.0);
}

#[test]
fn test_mlp_size() {
    let model = MLP::new(4, vec![10, 4, 1]);
    let parameters = model.parameters();
    assert_eq!(parameters.len(), 4 * 10 + 10 * 4 + 4 * 1 + 10 + 4 + 1);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_categorical_crossentropy() {
    microgradr::set_seed(1337);
    let inputs = vec![
        v1d![1.0, 2.0, 3.0, 4.0],
        v1d![5.0, 3.0, 0.0, 0.0],
        v1d![4.0, 2.0, 9.0, 4.0],
        v1d![1.0, 1.0, 1.0, 8.0],
    ];
    let targets = vec![0, 1, 2, 3];
    let model = MLP::new(4, vec![10, 4, 4]);
    let loss = categorical_crossentropy(&model, inputs, targets, 0.05, 100, 4);
    assert!((0.5159497443503941 - loss).abs() < 1e-6);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_binary_crossentropy() {
    microgradr::set_seed(1337);
    let inputs = vec![
        v1d![1.0, 2.0, 3.0, 4.0],
        v1d![5.0, 3.0, 0.0, 0.0],
        v1d![4.0, 2.0, 9.0, 4.0],
        v1d![1.0, 1.0, 1.0, 8.0],
    ];
    let targets = vec![false, true, false, true];
    let model = MLP::new(4, vec![10, 4, 1]);
    let loss = binary_crossentropy(&model, inputs.clone(), targets, 0.05, 100, 4);
    assert!((0.029123930840048218 - loss).abs() < 1e-6);

    let output = model.forward(inputs[0].clone());
    assert_eq!(output.len(), 1);
    assert!(output[0].data() < 0.0);
    let output = model.forward(inputs[1].clone());
    assert!(output[0].data() > 0.0);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_mean_squared_error() {
    microgradr::set_seed(1337);
    let inputs = vec![
        v1d![1.0, 2.0, 3.0, 4.0],
        v1d![5.0, 3.0, 0.0, 0.0],
        v1d![4.0, 2.0, 9.0, 4.0],
        v1d![1.0, 1.0, 1.0, 8.0],
    ];
    let targets = vec![v1d!(3.0), v1d!(1.0), v1d!(4.0), v1d!(9.0)];
    let model = MLP::new(4, vec![10, 4, 1]);
    let loss = mean_squared_error(&model, inputs, targets, 0.05, 100, 4);
    assert!((8.68750001573239 - loss).abs() < 1e-6);
}
