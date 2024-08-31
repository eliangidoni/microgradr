use microgradr::{
    mean_squared_error,
    sgd::{binary_crossentropy, categorical_crossentropy},
    v, vs, vvec, ValueVec, MLP,
};
use serial_test;

#[test]
fn test_engine_ops() {
    let x = v!(-4.0);
    let ref z = v!(2.0) * x + v!(2.0) + x;
    let ref q = z.relu() + z * x;
    let ref h = (z * z).relu();
    let y = h + q + q * x;
    y.backward();
    assert_eq!(x.grad(), 46.0);
    assert_eq!(y.data(), -20.0);

    let x = v!(-4.0);
    let y = x * x * v!(2.0) + x;
    y.backward();
    assert_eq!(y.data(), 28.0);
    assert_eq!(x.grad(), -15.0);

    let x = v!(0.9213);
    assert!((x.sigmoid().data() - 0.7153).abs() < 1e-4);
    let x = v!(1.0887);
    assert!((x.sigmoid().data() - 0.7481).abs() < 1e-4);
    let x = v!(-0.8858);
    assert!((x.sigmoid().data() - 0.2920).abs() < 1e-4);
    let x = v!(-1.7683);
    assert!((x.sigmoid().data() - 0.1458).abs() < 1e-4);

    let x = vvec!(-1.0, 2.0, -3.0);
    let y: microgradr::Value = x.iter().sum();
    assert!(y.data() == -2.0);
}

#[test]
fn test_engine_with_names() {
    let ref a = v!(1.0).with_name("a".to_string());
    let ref b = v!(2.0).with_name("b".to_string());
    let ref d = (a * a).with_name("a * a".to_string());
    let d = (d + (d * b).with_name("d * 2".to_string())).with_name("d + d * 2".to_string());
    d.backward();
    assert_eq!(d.data(), 3.0);
    assert_eq!(a.grad(), 6.0);

    let ref a = v!(1.0).with_name("a".to_string());
    let ref b = v!(2.0).with_name("b".to_string());
    let ref d = (a * a).with_name("a * a".to_string());
    let d = ((d * b).with_name("d * 2".to_string()) + d).with_name("d * d + 2".to_string());
    d.backward();
    assert_eq!(d.data(), 3.0);
    assert_eq!(a.grad(), 6.0);
}

#[test]
fn test_engine_more_ops() {
    let a = v!(-4.0);
    let b = v!(2.0);
    let ref c = a + b;
    let ref d = a * b + b.pow(v!(3.0));
    let ref c = c + c + v!(1.0);
    let ref c = c + v!(1.0) + c + (-a);
    let ref d = d + d * v!(2.0) + (b + a).relu();
    let ref d = d + v!(3.0) * d + (b - a).relu();
    let ref e = c - d;
    let ref f = e.pow(v!(2.0));
    let ref g = f / v!(2.0);
    let g = g + v!(10.0) / f;
    g.backward();

    let err = 1e-6;
    assert!((g.data() - 24.70408163265306).abs() < err);
    assert!((a.grad() - 138.83381924198252).abs() < err);
    assert!((b.grad() - 645.5772594752186).abs() < err);
}

#[test]
fn test_engine_relu() {
    let d = v!(-4.0);
    let a = v!(-4.0);
    let b = v!(250.0);
    let d = d + d * v!(2.0) + (b + a).relu();
    d.backward();
    assert_eq!(234.0, d.data());
    assert_eq!(1.0, a.grad());
    assert_eq!(1.0, b.grad());
}

#[test]
fn test_engine_self_ops() {
    let x = v!(3.0);
    let y = x + x;
    y.backward();
    assert_eq!(6.0, y.data());
    assert_eq!(2.0, x.grad());

    let x = v!(2.0);
    let y = x * x * (x * x);
    y.backward();
    assert_eq!(16.0, y.data());
    assert_eq!(32.0, x.grad());

    let x = v!(2.0);
    let y = x * x * x * x;
    y.backward();
    assert_eq!(16.0, y.data());
    assert_eq!(32.0, x.grad());

    let x = v!(2.0);
    let y = x + x + (x + x);
    y.backward();
    assert_eq!(8.0, y.data());
    assert_eq!(4.0, x.grad());

    let x = v!(2.0);
    let y = x + x + x + x;
    y.backward();
    assert_eq!(8.0, y.data());
    assert_eq!(4.0, x.grad());

    let x = v!(2.0);
    let y = x - x - (x - x);
    y.backward();
    assert_eq!(0.0, y.data());
    assert_eq!(0.0, x.grad());

    let x = v!(2.0);
    let y = x - x - x - x;
    y.backward();
    assert_eq!(-4.0, y.data());
    assert_eq!(-2.0, x.grad());

    let x = v!(2.0);
    let y = x.pow(v!(2.0)) * x.pow(v!(3.0)) * x.pow(v!(3.0)) * x.pow(v!(3.0));
    y.backward();
    assert_eq!(2048.0, y.data());
    assert_eq!(11264.0, x.grad());

    let x = v!(2.0);
    let y = x.pow(v!(2.0)) * x.pow(v!(3.0)) * (x.pow(v!(3.0)) * x.pow(v!(3.0)));
    y.backward();
    assert_eq!(2048.0, y.data());
    assert_eq!(11264.0, x.grad());

    let x = v!(4.0);
    let y = x / x / x / x;
    y.backward();
    assert_eq!(0.0625, y.data());
    assert_eq!(-0.03125, x.grad());

    let x = v!(4.0);
    let y = x / x / (x / x);
    y.backward();
    assert_eq!(1.0, y.data());
    assert_eq!(0.0, x.grad());

    let x = v!(4.0);
    let y = x.relu() * x.relu() * x.relu() * x.relu();
    y.backward();
    assert_eq!(256.0, y.data());
    assert_eq!(256.0, x.grad());

    let x = v!(4.0);
    let y = x.relu() * x.relu() * (x.relu() * x.relu());
    y.backward();
    assert_eq!(256.0, y.data());
    assert_eq!(256.0, x.grad());

    let x = v!(4.0);
    let y = x.tanh() * x.tanh() * x.tanh() * x.tanh();
    y.backward();
    assert!((0.9973198967826825 - y.data()).abs() < 1e-6);
    assert!((0.005353017457349722 - x.grad()).abs() < 1e-6);

    let x = v!(4.0);
    let y = x.tanh() * x.tanh() * (x.tanh() * x.tanh());
    y.backward();
    assert!((0.9973198967826825 - y.data()).abs() < 1e-6);
    assert!((0.005353017457349722 - x.grad()).abs() < 1e-6);

    let x = v!(4.0);
    let y = x.sigmoid() * x.sigmoid() * (x.sigmoid() * x.sigmoid());
    y.backward();
    assert!((0.9299728870391846 - y.data()).abs() < 1e-6);
    assert!((0.06690686196088791 - x.grad()).abs() < 1e-6);

    let x = v!(256.0);
    let y = x.log10() * x.log10() * (x.log10() * x.log10());
    y.backward();
    assert!((33.63559341430664 - y.data()).abs() < 1e-4);
    assert!((0.09477715194225311 - x.grad()).abs() < 1e-6);

    let x = v!(256.0);
    let y = x.log() * x.log() * (x.log() * x.log());
    y.backward();
    assert!((945.5005493164062 - y.data()).abs() < 1e-4);
    assert!((2.6641972064971924 - x.grad()).abs() < 1e-6);

    let x = v!(2.0);
    let y = x.exp() * x.exp();
    y.backward();
    assert!((54.59815216064453 - y.data()).abs() < 1e-4);
    assert!((109.19630432128906 - x.grad()).abs() < 1e-4);

    let x = v!(4.0);
    let y = x * (-x) + (-x);
    y.backward();
    assert_eq!(-20.0, y.data());
    assert_eq!(-9.0, x.grad());
}

#[test]
#[serial_test::serial(with_seed)]
fn test_categorical_crossentropy() {
    microgradr::set_seed(1337);
    let inputs = vec![
        vs![1.0, 2.0, 3.0, 4.0],
        vs![5.0, 3.0, 0.0, 0.0],
        vs![4.0, 2.0, 9.0, 4.0],
        vs![1.0, 1.0, 1.0, 8.0],
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
        vs![1.0, 2.0, 3.0, 4.0],
        vs![5.0, 3.0, 0.0, 0.0],
        vs![4.0, 2.0, 9.0, 4.0],
        vs![1.0, 1.0, 1.0, 8.0],
    ];
    let targets = vec![false, true, false, true];
    let model = MLP::new(4, vec![10, 4, 1]);
    let loss = binary_crossentropy(&model, inputs, targets, 0.05, 100, 4);
    assert!((0.029123930840048218 - loss).abs() < 1e-6);
}

#[test]
#[serial_test::serial(with_seed)]
fn test_mean_squared_error() {
    microgradr::set_seed(1337);
    let inputs = vec![
        vs![1.0, 2.0, 3.0, 4.0],
        vs![5.0, 3.0, 0.0, 0.0],
        vs![4.0, 2.0, 9.0, 4.0],
        vs![1.0, 1.0, 1.0, 8.0],
    ];
    let targets = vec![vs!(3.0), vs!(1.0), vs!(4.0), vs!(9.0)];
    let model = MLP::new(4, vec![10, 4, 1]);
    let loss = mean_squared_error(&model, inputs, targets, 0.05, 100, 4);
    assert!((8.68750001573239 - loss).abs() < 1e-6);
}

#[test]
fn test_value_vec() {
    let values = ValueVec::from(vec![1.0, 2.0, 3.0]);
    assert_eq!(values.len(), 3);
    assert_eq!(values[0].data(), 1.0);
    assert_eq!(values[1].data(), 2.0);
    assert_eq!(values[2].data(), 3.0);

    let x = vs!(-1.0, 2.0, -3.0);
    let y: microgradr::Value = x.iter().sum();
    assert!(y.data() == -2.0);

    let ref x = vs!(-1.0, 2.0, -3.0);
    let ref y = vs!(2.0, 3.0, -4.0);
    let y = x * y;
    y.backward();
    assert_eq!(vec![-2.0, 6.0, 12.0], y.data());
    assert_eq!(vec![2.0, 3.0, -4.0], x.grad());

    let x = vs!(-1.0, 2.0, -3.0);
    let y = vs!(2.0, 3.0, -4.0);
    let y = x + y;
    assert_eq!(vec![1.0, 5.0, -7.0], y.data());

    let x = vs!(-1.0, 2.0, -3.0);
    let y = vs!(2.0, 3.0, -4.0);
    let y = x - y;
    assert_eq!(vec![-3.0, -1.0, 1.0], y.data());

    let x = vs!(-1.0, 2.0, -12.0);
    let y = vs!(2.0, 2.0, -4.0);
    let y = x / y;
    assert_eq!(vec![-0.5, 1.0, 3.0], y.data());

    let x = vs!(-1.0, 2.0, -3.0);
    let y = v!(2.0);
    let y = x + y;
    assert_eq!(vec![1.0, 4.0, -1.0], y.data());

    let x = vs!(-1.0, 2.0, -3.0);
    let y = v!(-2.0);
    let y = x - y;
    assert_eq!(vec![1.0, 4.0, -1.0], y.data());

    let x = vs!(-1.0, 2.0, -3.0);
    let y = v!(-2.0);
    let y = y - x;
    assert_eq!(vec![-1.0, -4.0, 1.0], y.data());

    let x = vs!(-1.0, 2.0, -3.0);
    let y = v!(2.0);
    let y = x * y;
    assert_eq!(vec![-2.0, 4.0, -6.0], y.data());

    let x = vs!(-1.0, 2.0, -5.0);
    let y = v!(2.0);
    let y = x / y;
    assert_eq!(vec![-0.5, 1.0, -2.5], y.data());

    let x = vs!(-1.0, 2.0, -5.0);
    let y = v!(2.0);
    let y = y / x;
    assert_eq!(vec![-2.0, 1.0, -0.4], y.data());

    let ref x = vs!(-12.0, 1.0, 23.0);
    let xx = v!(3.0);
    let y = xx * x.mean();
    y.backward();
    assert_eq!(12.0, y.data());
    assert_eq!(1.0, x.grad()[0]);
    assert_eq!(1.0, x.grad()[1]);
    assert_eq!(1.0, x.grad()[2]);
    assert_eq!(4.0, xx.grad());

    let x = vs!(-1.0, 2.0, -2.0);
    let y = vs!(2.0, 3.0, -2.0);
    let y = x.pow(&y);
    assert_eq!(vec![1.0, 8.0, 0.25], y.data());

    let ref x = vs!(4.0, 4.0, 2.0);
    let ref y = x * vs!(1.0, 2.0, 3.0);
    let y = y.softmax().log();
    y.backward();
    assert!((-4.1429 - y.data()[0]).abs() < 1e-4);
    assert!((-0.1429 - y.data()[1]).abs() < 1e-4);
    assert!((-2.1429 - y.data()[2]).abs() < 1e-4);
    assert!((0.9524 - x.grad()[0]).abs() < 1e-4);
    assert!((-3.2009 - x.grad()[1]).abs() < 1e-4);
    assert!((1.9442 - x.grad()[2]).abs() < 1e-4);

    let ref x = vs!(3.0, 2.0, 3.0);
    let ref y = vs!(2.0, 4.0, 5.0);
    let y = x * y.max();
    y.backward();
    assert_eq!(vec![15.0, 10.0, 15.0], y.data());
    assert_eq!(vec![5.0, 5.0, 5.0], x.grad());

    let ref x = vs!(3.0, 2.0, 3.0);
    let ref y = vs!(2.0, 4.0, 5.0);
    let y = x * y.min();
    y.backward();
    assert_eq!(vec![6.0, 4.0, 6.0], y.data());
    assert_eq!(vec![2.0, 2.0, 2.0], x.grad());

    let ref x = vs!(3.0, 2.0, 3.0);
    let ref y = vs!(2.0, 4.0, 5.0);
    let (y, idx) = y.max_index();
    let y = x * y;
    y.backward();
    assert_eq!(2.0, idx.data());
    assert_eq!(vec![15.0, 10.0, 15.0], y.data());
    assert_eq!(vec![5.0, 5.0, 5.0], x.grad());

    let ref x = vs!(3.0, 2.0, 3.0);
    let ref y = vs!(2.0, 4.0, 5.0);
    let (y, idx) = y.min_index();
    let y = x * y;
    y.backward();
    assert_eq!(0.0, idx.data());
    assert_eq!(vec![6.0, 4.0, 6.0], y.data());
    assert_eq!(vec![2.0, 2.0, 2.0], x.grad());

    assert!(vs!(3.0, 2.0, -3.0) == vs!(3.0, 2.0, -3.0));
    assert!(vs!(3.0, 2.0, -3.0) != vs!(3.0, 2.0, 3.0));
    assert!(vs!(3.0, 2.0, -3.0) < vs!(4.0, 3.0, -2.0));
    assert!(vs!(5.0, 4.0, 0.0) > vs!(4.0, 3.0, -2.0));
    assert!(vs!(3.0, 2.0, 3.0) == vs!(3, 2, 3));
    assert!(vs!(1.0, 1.0, 0.0) == vs!(true, true, false));
}
