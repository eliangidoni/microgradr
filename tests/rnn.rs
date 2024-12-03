use microgradr::{
    sgd::{categorical_crossentropy_lstm, categorical_crossentropy_rnn}, v1d, DropoutLayer, LstmLayer, RecurrentLayer, Value, LSTM,
    RNN,
};

#[test]
fn test_dropout_layer() {
    let input = v1d!(1.0, 2.0);
    let layer = DropoutLayer::new(0.0);
    let output = layer.forward(&input);
    assert_eq!(output, v1d![1.0, 2.0]);

    let input = v1d!(1.0, 2.0);
    let layer = DropoutLayer::new(1.0);
    let output = layer.forward(&input);
    assert_eq!(output, v1d![0.0, 0.0]);
}

#[test]
fn test_recurrent_layer() {
    let input = v1d!(1.0, 3.0);
    let mut layer = RecurrentLayer::new(2, 2, 0.0);
    layer.set_weights(Value::from(1.0));
    layer.set_bias(Value::from(0.0));
    let (output, _) = layer.forward(&input, None);
    assert!(output[0].data() - 0.999329299739067 < 1e-6);
    assert!(output[1].data() - 0.999329299739067 < 1e-6);

    let input = v1d!(0.1, 0.3);
    let mut layer = RecurrentLayer::new(2, 2, 0.0);
    layer.set_weights(Value::from(1.0));
    layer.set_bias(Value::from(1.0));
    let (output, state) = layer.forward(&input, None);
    assert!(output[0].data() - 0.9836748576936802 < 1e-6);
    assert!(output[1].data() - 0.9836748576936802 < 1e-6);

    let input = v1d!(0.05, 0.05);
    let (output, _) = layer.forward(&input, state);
    assert!(output[0].data() - 0.9994137975758832 < 1e-6);
    assert!(output[1].data() - 0.9994137975758832 < 1e-6);
}

#[test]
fn test_rnn() {
    let input = vec![v1d!(0.1, 0.3)];
    let mut rnn = RNN::new(2, 2, 2, 1, 0.0);
    rnn.set_weights(Value::from(1.0));
    rnn.set_bias(Value::from(0.0));
    let (output, state) = rnn.forward(&input, None);
    assert!(output[0].data()[0] - 0.9836748576936802 < 1e-6);
    assert!(output[0].data()[1] - 0.9836748576936802 < 1e-6);

    let states = state.unwrap();
    assert_eq!(states.len(), 1);
    assert_eq!(states[0].data().len(), 2);
    assert!(states[0].data()[0] - 0.3799489622552249 < 1e-6);
    assert!(states[0].data()[1] - 0.3799489622552249 < 1e-6);
}

#[test]
fn test_lstm_layer() {
    let input = v1d!(1.0, 3.0);
    let mut layer = LstmLayer::new(2, 2, 0.0);
    layer.set_weights(Value::from(1.0));
    layer.set_bias(Value::from(0.0));
    let (output, state) = layer.forward(&input, None);
    assert!(output[0].data() - 0.7400965990008488 < 1e-6);
    assert!(output[1].data() - 0.7400965990008488 < 1e-6);

    let (h, c) = state.unwrap();
    assert!(h[0].data() - 0.7400965990008488 < 1e-6);
    assert!(h[1].data() - 0.7400965990008488 < 1e-6);
    assert!(c[0].data() - 0.9813551531326903 < 1e-6);
    assert!(c[1].data() - 0.9813551531326903 < 1e-6);

    let input = v1d!(0.1, 0.3);
    let mut layer = LstmLayer::new(2, 2, 0.0);
    layer.set_weights(Value::from(1.0));
    layer.set_bias(Value::from(1.0));
    let (output, state) = layer.forward(&input, None);
    assert!(output[0].data() - 0.6575506641133497 < 1e-6);
    assert!(output[1].data() - 0.6575506641133497 < 1e-6);

    let (h, c) = state.clone().unwrap();

    assert!(h[0].data() - 0.6575506641133497 < 1e-6);
    assert!(h[1].data() - 0.6575506641133497 < 1e-6);
    assert!(c[0].data() - 0.9018599673060215 < 1e-6);
    assert!(c[1].data() - 0.9018599673060215 < 1e-6);

    let input = v1d!(0.05, 0.05);
    let (output, _) = layer.forward(&input, state);
    assert!(output[0].data() - 0.9204636265819589 < 1e-6);
    assert!(output[1].data() - 0.9204636265819589 < 1e-6);
}

#[test]
fn test_lstm() {
    let input = vec![v1d!(0.1, 0.3)];
    let mut lstm = LSTM::new(2, 2, 2, 1, 0.0);
    lstm.set_weights(Value::from(1.0));
    lstm.set_bias(Value::from(0.0));
    let (output, state) = lstm.forward(&input, None);
    assert!(output[0].data()[0] - 0.9836748576936802 < 1e-6);
    assert!(output[0].data()[1] - 0.9836748576936802 < 1e-6);

    let states = state.unwrap();
    assert_eq!(states.len(), 1);
    assert_eq!(states[0].0.data().len(), 2);
    assert_eq!(states[0].1.data().len(), 2);
    assert!(states[0].0[0].data() - 0.1338826989447839 < 1e-6);
    assert!(states[0].0[1].data() - 0.1338826989447839 < 1e-6);
    assert!(states[0].1[0].data() - 0.22747075517473495 < 1e-6);
    assert!(states[0].1[1].data() - 0.22747075517473495 < 1e-6);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_categorical_crossentropy_rnn() {
    microgradr::set_seed(1337);
    let input = vec![
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![0.0, 1.0, 0.0, 1.0, 0.0],
        v1d![0.0, 0.0, 1.0, 0.0, 0.0],
        v1d![0.0, 1.0, 0.0, 1.0, 0.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
    ];
    let input2 = vec![
        v1d![1.0, 1.0, 1.0, 1.0, 1.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![1.0, 1.0, 1.0, 1.0, 1.0],
    ];
    let inputs = vec![input.clone(), input2.clone()];
    let targets = vec![vec![0, 0, 0, 0, 0], vec![1, 1, 1, 1, 1]];
    let mut model = RNN::new(input[0].len(), input[0].len(), 2, 1, 0.0);
    let loss = categorical_crossentropy_rnn(&model, inputs, targets, 0.5, 100, 1);
    assert!((0.010845145299080993 - loss).abs() < 1e-6);

    model.eval();
    let output = model
        .forward(&input, None)
        .0
        .iter()
        .map(|x| x.argmax())
        .collect::<Vec<usize>>();
    assert_eq!(output, vec![0, 0, 0, 0, 0]);
    let output = model
        .forward(&input2, None)
        .0
        .iter()
        .map(|x| x.argmax())
        .collect::<Vec<usize>>();
    assert_eq!(output, vec![1, 1, 1, 1, 1]);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_categorical_crossentropy_lstm() {
    microgradr::set_seed(1337);
    let input = vec![
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![0.0, 1.0, 0.0, 1.0, 0.0],
        v1d![0.0, 0.0, 1.0, 0.0, 0.0],
        v1d![0.0, 1.0, 0.0, 1.0, 0.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
    ];
    let input2 = vec![
        v1d![1.0, 1.0, 1.0, 1.0, 1.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![1.0, 0.0, 0.0, 0.0, 1.0],
        v1d![1.0, 1.0, 1.0, 1.0, 1.0],
    ];
    let inputs = vec![input.clone(), input2.clone()];
    let targets = vec![vec![0, 0, 0, 0, 0], vec![1, 1, 1, 1, 1]];
    let mut model = LSTM::new(input[0].len(), input[0].len(), 2, 1, 0.0);
    let loss = categorical_crossentropy_lstm(&model, inputs, targets, 0.5, 200, 1);
    assert!((0.06210688982730858 - loss).abs() < 1e-6);

    model.eval();
    let output = model
        .forward(&input, None)
        .0
        .iter()
        .map(|x| x.argmax())
        .collect::<Vec<usize>>();
    assert_eq!(output, vec![0, 0, 0, 0, 0]);
    let output = model
        .forward(&input2, None)
        .0
        .iter()
        .map(|x| x.argmax())
        .collect::<Vec<usize>>();
    assert_eq!(output, vec![1, 1, 1, 1, 1]);
}
