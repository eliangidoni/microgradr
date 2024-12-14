use microgradr::{
    v1d, Attention, Decoder, DecoderLayer, Embedding, Encoder, EncoderLayer, FeedForward,
    LayerNorm, MultiheadAttention, PositionalEncoding, Transformer,
};

#[test]
#[serial_test::serial(with_seed)]
fn test_embedding() {
    microgradr::set_seed(1337);
    let input = vec![0, 1];
    let layer = Embedding::new(2, 2);
    let output = layer.forward(&input);
    assert_eq!(output.len(), 2);
    assert_eq!(output[0].len(), 2);

    assert!(output[0].data()[0] - (-0.3561614426278048) < 1e-6);
    assert!(output[0].data()[1] - 1.0724867803987979 < 1e-6);
    assert!(output[1].data()[0] - (-1.121562763364784) < 1e-6);
    assert!(output[1].data()[1] - (-1.3184935837008842) < 1e-6);
}

#[test]
fn test_layernorm() {
    let input = vec![v1d![1.0, 2.0, 3.0], v1d![4.0, 5.0, 6.0]];
    let layer = LayerNorm::new(3, 1e-05);
    let output = layer.forward(&input);
    assert_eq!(output.len(), 2);
    assert_eq!(output[0].len(), 3);

    assert!(output[0].data()[0] - (-1.2247) < 1e-4);
    assert!(output[0].data()[1] - 0.0000 < 1e-4);
    assert!(output[0].data()[2] - (1.2247) < 1e-4);
    assert!(output[1].data()[0] - (-1.2247) < 1e-4);
    assert!(output[1].data()[1] - 0.0000 < 1e-4);
    assert!(output[1].data()[2] - (1.2247) < 1e-4);
}

#[test]
fn test_positional_encoding() {
    let layer = PositionalEncoding::new(4, 2);
    let input = vec![v1d![1.0, 2.0, 3.0, 4.0], v1d![5.0, 6.0, 7.0, 8.0]];
    let output = layer.forward(&input);
    assert_eq!(output.len(), 2);
    assert_eq!(output[0].len(), 4);

    assert!(output[0].data()[0] - 1.0000 < 1e-4);
    assert!(output[0].data()[1] - 3.0000 < 1e-4);
    assert!(output[0].data()[2] - 3.0000 < 1e-4);
    assert!(output[0].data()[3] - 5.0000 < 1e-4);
    assert!(output[1].data()[0] - 5.8414 < 1e-4);
    assert!(output[1].data()[1] - 6.9950 < 1e-4);
    assert!(output[1].data()[2] - 7.0099 < 1e-4);
    assert!(output[1].data()[3] - 8.9999 < 1e-4);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_feedforward() {
    microgradr::set_seed(1337);
    let input = vec![v1d![1.0, 2.0, 3.0], v1d![4.0, 5.0, 6.0]];
    let layer = FeedForward::new(3, 2);
    let output = layer.forward(&input);
    assert_eq!(output.len(), 2);
    assert_eq!(output[0].len(), 3);

    assert!((output[0].data()[0] - (-0.005426605456108372)).abs() < 1e-6);
    assert!((output[0].data()[1] - 0.25009104566904466).abs() < 1e-6);
    assert!((output[0].data()[2] - (-0.14947101108799016)).abs() < 1e-6);
    assert!((output[1].data()[0] - 0.0).abs() < 1e-6);
    assert!((output[1].data()[1] - 0.0).abs() < 1e-6);
    assert!((output[1].data()[2] - 0.0).abs() < 1e-6);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_attention() {
    microgradr::set_seed(1337);
    let input = vec![v1d![1.0, 2.0, 3.0, 4.0], v1d![5.0, 6.0, 7.0, 8.0]];
    let attn = Attention::new(4, 2, 0.0);
    let query = &input;
    let key = &input;
    let value = &input;
    let mask = None;
    let output = attn.forward(query, key, value, mask);
    assert_eq!(output.0.len(), 2);
    assert_eq!(output.0[0].len(), 2);

    assert!((output.0[0].data()[0] - 2.321519764411311).abs() < 1e-6);
    assert!((output.0[0].data()[1] - (-3.940793612418122)).abs() < 1e-6);
    assert!((output.0[1].data()[0] - 2.557591349530205).abs() < 1e-6);
    assert!((output.0[1].data()[1] - (-4.238969755847536)).abs() < 1e-6);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_multihead_attention() {
    microgradr::set_seed(1337);
    let input = vec![v1d![1.0, 2.0, 3.0, 4.0], v1d![5.0, 6.0, 7.0, 8.0]];
    let attn = MultiheadAttention::new(4, 2, 0.0);
    let query = &input;
    let key = &input;
    let value = &input;
    let mask = None;
    let output = attn.forward(query, key, value, mask);
    assert_eq!(output.0.len(), 2);
    assert_eq!(output.0[0].len(), 4);

    assert!((output.0[0].data()[0] - (-3.8998748851087597)).abs() < 1e-6);
    assert!((output.0[0].data()[1] - (-6.025598888310073)).abs() < 1e-6);
    assert!((output.0[0].data()[2] - (-3.0105274853375685)).abs() < 1e-6);
    assert!((output.0[0].data()[3] - 5.204654466615348).abs() < 1e-6);
    assert!((output.0[1].data()[0] - (-4.259470329973722)).abs() < 1e-6);
    assert!((output.0[1].data()[1] - (-6.02860748938458)).abs() < 1e-6);
    assert!((output.0[1].data()[2] - (-3.401331079041893)).abs() < 1e-6);
    assert!((output.0[1].data()[3] - 5.255154689449494).abs() < 1e-6);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_encoder_layer() {
    microgradr::set_seed(1337);
    let input = vec![v1d![1.0, 2.0, 3.0, 4.0], v1d![5.0, 6.0, 7.0, 8.0]];
    let layer = EncoderLayer::new(4, 2, 2, 0.0, 1e-05);
    let output = layer.forward(&input, None);
    assert_eq!(output.len(), 2);
    assert_eq!(output[0].len(), 4);

    assert!((output[0].data()[0] - (-0.6670623511594618)).abs() < 1e-6);
    assert!((output[0].data()[1] - (-0.8836530143750759)).abs() < 1e-6);
    assert!((output[0].data()[2] - (-0.11114840795748845)).abs() < 1e-6);
    assert!((output[0].data()[3] - 1.6618637734920263).abs() < 1e-6);
    assert!((output[1].data()[0] - (-0.6897686818864126)).abs() < 1e-6);
    assert!((output[1].data()[1] - (-0.8350823688509618)).abs() < 1e-6);
    assert!((output[1].data()[2] - (-0.14977819081570398)).abs() < 1e-6);
    assert!((output[1].data()[3] - 1.6746292415530786).abs() < 1e-6);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_decoder_layer() {
    microgradr::set_seed(1337);
    let input = vec![v1d![1.0, 2.0, 3.0, 4.0], v1d![5.0, 6.0, 7.0, 8.0]];
    let layer = DecoderLayer::new(4, 2, 2, 0.0, 1e-05);
    let output = layer.forward(&input, &input, None, None);
    assert_eq!(output.len(), 2);
    assert_eq!(output[0].len(), 4);

    assert!((output[0].data()[0] - (-1.13511945641133)).abs() < 1e-6);
    assert!((output[0].data()[1] - 0.08466667770803875).abs() < 1e-6);
    assert!((output[0].data()[2] - 1.5626663600610167).abs() < 1e-6);
    assert!((output[0].data()[3] - (-0.5122135813577254)).abs() < 1e-6);
    assert!((output[1].data()[0] - (-1.1374016399328135)).abs() < 1e-6);
    assert!((output[1].data()[1] - 0.08852426001037086).abs() < 1e-6);
    assert!((output[1].data()[2] - 1.5608658389496228).abs() < 1e-6);
    assert!((output[1].data()[3] - (-0.5119884590271803)).abs() < 1e-6);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_encoder() {
    microgradr::set_seed(1337);
    let input = vec![v1d![1.0, 2.0, 3.0, 4.0], v1d![5.0, 6.0, 7.0, 8.0]];
    let layer = EncoderLayer::new(4, 2, 2, 0.0, 1e-05);
    let encoder = Encoder::new(layer, 2);
    let output = encoder.forward(&input, None);
    assert_eq!(output.len(), 2);
    assert_eq!(output[0].len(), 4);

    assert!((output[0].data()[0] - (-0.6670623511594618)).abs() < 1e-6);
    assert!((output[0].data()[1] - (-0.8836530143750759)).abs() < 1e-6);
    assert!((output[0].data()[2] - (-0.11114840795748845)).abs() < 1e-6);
    assert!((output[0].data()[3] - 1.6618637734920263).abs() < 1e-6);
    assert!((output[1].data()[0] - (-0.6897686818864126)).abs() < 1e-6);
    assert!((output[1].data()[1] - (-0.8350823688509618)).abs() < 1e-6);
    assert!((output[1].data()[2] - (-0.14977819081570398)).abs() < 1e-6);
    assert!((output[1].data()[3] - 1.6746292415530786).abs() < 1e-6);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_decoder() {
    microgradr::set_seed(1337);
    let input = vec![v1d![1.0, 2.0, 3.0, 4.0], v1d![5.0, 6.0, 7.0, 8.0]];
    let layer = DecoderLayer::new(4, 2, 2, 0.0, 1e-05);
    let decoder = Decoder::new(layer, 2);
    let output = decoder.forward(&input, &input, None, None);
    assert_eq!(output.len(), 2);
    assert_eq!(output[0].len(), 4);

    assert!((output[0].data()[0] - (-0.6796263723085445)).abs() < 1e-6);
    assert!((output[0].data()[1] - (-0.31702485435766964)).abs() < 1e-6);
    assert!((output[0].data()[2] - 1.7109501817781376).abs() < 1e-6);
    assert!((output[0].data()[3] - (-0.7142989551119233)).abs() < 1e-6);
    assert!((output[1].data()[0] - (-0.6796159966563516)).abs() < 1e-6);
    assert!((output[1].data()[1] - (-0.3169396085517471)).abs() < 1e-6);
    assert!((output[1].data()[2] - 1.710936066334238).abs() < 1e-6);
    assert!((output[1].data()[3] - (-0.7143804611261393)).abs() < 1e-6);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_transformer() {
    microgradr::set_seed(1337);
    let src = vec![v1d![1.0, 2.0, 3.0, 4.0], v1d![5.0, 6.0, 7.0, 8.0]];
    let tgt = vec![v1d![8.0, 7.0, 6.0, 5.0], v1d![4.0, 3.0, 2.0, 1.0]];
    let transformer = Transformer::new(4, 2, 2, 2, 2, 0.0, 1e-05);
    let output = transformer.forward(&src, &tgt, None, None, None);
    assert_eq!(output.len(), 2);
    assert_eq!(output[0].len(), 4);

    assert!((output[0].data()[0] - (-0.7136391519555993)).abs() < 1e-6);
    assert!((output[0].data()[1] - 1.5305530809764825).abs() < 1e-6);
    assert!((output[0].data()[2] - (-1.0465840066106529)).abs() < 1e-6);
    assert!((output[0].data()[3] - 0.22967007758976957).abs() < 1e-6);
    assert!((output[1].data()[0] - (-0.7138219956159537)).abs() < 1e-6);
    assert!((output[1].data()[1] - 1.52640907010125).abs() < 1e-6);
    assert!((output[1].data()[2] - (-1.0506322031634572)).abs() < 1e-6);
    assert!((output[1].data()[3] - 0.23804512867816083).abs() < 1e-6);
}
