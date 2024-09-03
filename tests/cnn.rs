use microgradr::{
    sgd::{binary_crossentropy_2d, categorical_crossentropy_2d},
    v1d, Conv2dLayer, Dropout2dLayer, MaxPooling2dLayer, Value2d, CNN,
};

#[test]
fn test_maxpooling2d_layer() {
    let input = Value2d::from(vec![
        v1d![1.0, 2.0, 1.0, 5.0],
        v1d![0.0, 1.0, 2.0, 1.0],
        v1d![1.0, 0.0, 1.0, 3.0],
        v1d![9.0, 1.0, 0.0, 1.0],
    ]);
    let layer = MaxPooling2dLayer::new(2, 2);
    let output = layer.forward(&vec![input]);
    assert_eq!(
        output,
        vec![Value2d::from(vec![v1d![2.0, 5.0], v1d![9.0, 3.0]])]
    );
}

#[test]
fn test_dropout2d_layer() {
    let input = Value2d::from(vec![v1d![1.0, 2.0], v1d![0.0, 1.0]]);
    let layer = Dropout2dLayer::new(0.0);
    let output = layer.forward(&vec![input]);
    assert_eq!(
        output,
        vec![Value2d::from(vec![v1d![1.0, 2.0], v1d![0.0, 1.0]])]
    );

    let input = Value2d::from(vec![v1d![1.0, 2.0], v1d![0.0, 1.0]]);
    let layer = Dropout2dLayer::new(1.0);
    let output = layer.forward(&vec![input]);
    assert_eq!(
        output,
        vec![Value2d::from(vec![v1d![0.0, 0.0], v1d![0.0, 0.0]])]
    );
}

#[test]
fn test_conv2d_layer() {
    let input = Value2d::from(vec![
        v1d![1.0, 2.0, 1.0, 5.0],
        v1d![0.0, 1.0, 2.0, 10.0],
        v1d![1.0, 0.0, 1.0, 3.0],
        v1d![9.0, 1.0, 0.0, 9.0],
    ]);
    let weights = v1d![1.0, 0.0, 0.0, 1.0];
    let mut layer = Conv2dLayer::new(1, 1, 2, 2);
    layer.set_weights(weights);
    layer.set_biases(v1d![1.0]);
    let output = layer.forward(&vec![input]);
    let expected = vec![Value2d::from(vec![v1d![3.0, 12.0], v1d![3.0, 11.0]])];
    assert_eq!(output, expected);

    let input = Value2d::from(vec![
        v1d![1.0, 2.0, 1.0, 5.0],
        v1d![0.0, 1.0, 2.0, 10.0],
        v1d![1.0, 0.0, 1.0, 3.0],
        v1d![9.0, 1.0, 0.0, 9.0],
    ]);
    let input2 = Value2d::from(vec![
        v1d![-1.0, 2.0, -1.0, 5.0],
        v1d![0.0, -1.0, 2.0, -10.0],
        v1d![-1.0, 0.0, -1.0, 3.0],
        v1d![9.0, -1.0, 0.0, -9.0],
    ]);
    let weights = v1d![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let mut layer = Conv2dLayer::new(2, 1, 2, 2);
    layer.set_weights(weights.clone());
    layer.set_biases(v1d![11.0]);
    let output = layer.forward(&vec![input.clone(), input2.clone()]);
    let expected = vec![Value2d::from(vec![v1d![11.0, 11.0], v1d![11.0, 11.0]])];
    assert_eq!(output, expected);

    let mut layer = Conv2dLayer::new(2, 1, 2, 1);
    layer.set_weights(weights.clone());
    layer.set_biases(v1d![11.0]);
    let output = layer.forward(&vec![input.clone(), input2.clone()]);
    let expected = vec![Value2d::from(vec![
        v1d![11.0, 19.0, 11.0],
        v1d![11.0, 11.0, 21.0],
        v1d![11.0, 11.0, 11.0],
    ])];
    assert_eq!(output, expected);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_categorical_crossentropy_2d() {
    microgradr::set_seed(1337);
    let input = vec![Value2d::from(vec![
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![0.0, 1.0, 0.0, 1.0, 0.0],
        v1d![0.0, 0.0, 1.0, 0.0, 0.0],
        v1d![0.0, 1.0, 0.0, 1.0, 0.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
    ])];
    let input2 = vec![Value2d::from(vec![
        v1d![1.0, 1.0, 1.0, 1.0, 1.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![1.0, 1.0, 1.0, 1.0, 1.0],
    ])];
    let inputs = vec![input.clone(), input2.clone()];
    let targets = vec![0, 1];
    let mut model = CNN::new(1, input[0].shape().0, input[0].shape().1, 2);
    let loss = categorical_crossentropy_2d(&model, inputs, targets, 0.05, 100, 1);
    assert!((0.013941590085877108 - loss).abs() < 1e-6);

    model.eval();
    let output = model.forward(input).argmax();
    assert_eq!(output, 0);
    let output = model.forward(input2).argmax();
    assert_eq!(output, 1);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_binary_crossentropy_2d() {
    microgradr::set_seed(1337);
    let input = vec![Value2d::from(vec![
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![0.0, 1.0, 0.0, 1.0, 0.0],
        v1d![0.0, 0.0, 1.0, 0.0, 0.0],
        v1d![0.0, 1.0, 0.0, 1.0, 0.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
    ])];
    let input2 = vec![Value2d::from(vec![
        v1d![1.0, 1.0, 1.0, 1.0, 1.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![1.0, 1.0, 1.0, 1.0, 1.0],
    ])];
    let inputs = vec![input.clone(), input2.clone()];
    let targets = vec![false, true];
    let mut model = CNN::new(1, input[0].shape().0, input[0].shape().1, 1);
    let loss = binary_crossentropy_2d(&model, inputs, targets, 0.05, 100, 1);
    assert!((0.02140549357292931 - loss).abs() < 1e-6);

    model.eval();
    let output = model.forward(input);
    assert_eq!(1, output.len());
    assert!(output[0].data() < 0.0);
    let output = model.forward(input2);
    assert!(output[0].data() > 0.0);
}
