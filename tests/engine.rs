use microgradr::{v, v1d, v2d, Value, Value1d, Value2d};

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
    let y = x.pow(v!(2.0)) * x.pow(v!(3.0)).sqrt() * x.pow(v!(3.0)) * x.pow(v!(3.0));
    y.backward();
    assert!((y.data() - 724.0773).abs() < 1e-4);
    assert!((x.grad() - 3439.3674).abs() < 1e-4);

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
fn test_value_vec() {
    let values = Value1d::from(vec![1.0, 2.0, 3.0]);
    assert_eq!(values.len(), 3);
    assert_eq!(values[0].data(), 1.0);
    assert_eq!(values[1].data(), 2.0);
    assert_eq!(values[2].data(), 3.0);

    let x = v1d!(-1.0, 2.0, -3.0);
    let y: microgradr::Value = x.iter().sum();
    assert!(y.data() == -2.0);

    let ref x = v1d!(-1.0, 2.0, -3.0);
    let ref y = v1d!(2.0, 3.0, -4.0);
    let y = x * y;
    y.backward();
    assert_eq!(vec![-2.0, 6.0, 12.0], y.data());
    assert_eq!(vec![2.0, 3.0, -4.0], x.grad());

    let x = v1d!(-1.0, 2.0, -3.0);
    let y = v1d!(2.0, 3.0, -4.0);
    let y = x + y;
    assert_eq!(vec![1.0, 5.0, -7.0], y.data());

    let x = v1d!(-1.0, 2.0, -3.0);
    let y = v1d!(2.0, 3.0, -4.0);
    let y = x - y;
    assert_eq!(vec![-3.0, -1.0, 1.0], y.data());

    let x = v1d!(-1.0, 2.0, -12.0);
    let y = v1d!(2.0, 2.0, -4.0);
    let y = x / y;
    assert_eq!(vec![-0.5, 1.0, 3.0], y.data());

    let x = v1d!(-1.0, 2.0, -3.0);
    let y = v!(2.0);
    let y = x + y;
    assert_eq!(vec![1.0, 4.0, -1.0], y.data());

    let x = v1d!(-1.0, 2.0, -3.0);
    let y = v!(-2.0);
    let y = x - y;
    assert_eq!(vec![1.0, 4.0, -1.0], y.data());

    let x = v1d!(-1.0, 2.0, -3.0);
    let y = v!(-2.0);
    let y = y - x;
    assert_eq!(vec![-1.0, -4.0, 1.0], y.data());

    let x = v1d!(-1.0, 2.0, -3.0);
    let y = v!(2.0);
    let y = x * y;
    assert_eq!(vec![-2.0, 4.0, -6.0], y.data());

    let x = v1d!(-1.0, 2.0, -5.0);
    let y = v!(2.0);
    let y = x / y;
    assert_eq!(vec![-0.5, 1.0, -2.5], y.data());

    let x = v1d!(-1.0, 2.0, -5.0);
    let y = v!(2.0);
    let y = y / x;
    assert_eq!(vec![-2.0, 1.0, -0.4], y.data());

    let ref x = v1d!(-12.0, 1.0, 23.0);
    let xx = v!(3.0);
    let y = xx * x.mean();
    y.backward();
    assert_eq!(12.0, y.data());
    assert_eq!(1.0, x.grad()[0]);
    assert_eq!(1.0, x.grad()[1]);
    assert_eq!(1.0, x.grad()[2]);
    assert_eq!(4.0, xx.grad());

    let x = v1d!(-1.0, 2.0, -2.0);
    let y = v1d!(2.0, 3.0, -2.0);
    let y = x.pow(&y);
    assert_eq!(vec![1.0, 8.0, 0.25], y.data());

    let ref x = v1d!(4.0, 4.0, 2.0);
    let ref y = x * v1d!(1.0, 2.0, 3.0);
    let y = y.softmax().log();
    y.backward();
    assert!((-4.1429 - y.data()[0]).abs() < 1e-4);
    assert!((-0.1429 - y.data()[1]).abs() < 1e-4);
    assert!((-2.1429 - y.data()[2]).abs() < 1e-4);
    assert!((0.9524 - x.grad()[0]).abs() < 1e-4);
    assert!((-3.2009 - x.grad()[1]).abs() < 1e-4);
    assert!((1.9442 - x.grad()[2]).abs() < 1e-4);

    let ref x = v1d!(3.0, 2.0, 3.0);
    let ref y = v1d!(2.0, 4.0, 5.0);
    let y = x * y.max();
    y.backward();
    assert_eq!(vec![15.0, 10.0, 15.0], y.data());
    assert_eq!(vec![5.0, 5.0, 5.0], x.grad());

    let ref x = v1d!(3.0, 2.0, 3.0);
    let ref y = v1d!(2.0, 4.0, 5.0);
    let y = x * y.min();
    y.backward();
    assert_eq!(vec![6.0, 4.0, 6.0], y.data());
    assert_eq!(vec![2.0, 2.0, 2.0], x.grad());

    let ref x = v1d!(3.0, 2.0, 3.0);
    let ref y = v1d!(2.0, 4.0, 5.0);
    let (y, idx) = (y.max(), y.argmax());
    let y = x * y;
    y.backward();
    assert_eq!(2, idx);
    assert_eq!(vec![15.0, 10.0, 15.0], y.data());
    assert_eq!(vec![5.0, 5.0, 5.0], x.grad());

    let ref x = v1d!(3.0, 2.0, 3.0);
    let ref y = v1d!(2.0, 4.0, 5.0);
    let (y, idx) = (y.min(), y.argmin());
    let y = x * y;
    y.backward();
    assert_eq!(0, idx);
    assert_eq!(vec![6.0, 4.0, 6.0], y.data());
    assert_eq!(vec![2.0, 2.0, 2.0], x.grad());

    assert!(v1d!(3.0, 2.0, -3.0) == v1d!(3.0, 2.0, -3.0));
    assert!(v1d!(3.0, 2.0, -3.0) != v1d!(3.0, 2.0, 3.0));
    assert!(v1d!(3.0, 2.0, -3.0) < v1d!(4.0, 3.0, -2.0));
    assert!(v1d!(5.0, 4.0, 0.0) > v1d!(4.0, 3.0, -2.0));
    assert!(v1d!(3.0, 2.0, 3.0) == v1d!(3, 2, 3));
    assert!(v1d!(1.0, 1.0, 0.0) == v1d!(true, true, false));

    assert!(v!(1.0) == v!(1.0));
    assert!(v!(1.0) <= v!(1.5));
    assert!(v!(1.5) == v!(1.5));
    assert!(v!(1.0) < v!(1.5));
    assert!(v!(1.5) >= v!(1.0));
    assert!(v!(1.5) != v!(1.0));
    assert!(v!(1.5) > v!(1.0));

    let ref x = v1d!(3.0, 2.0, 3.0);
    let ref y = v1d!(2.0, 4.0, 5.0);
    assert_eq!(v!(29), &x.dot(y));

    let ref x = v1d!(3.0, 2.0, 3.0);
    let ref y = v1d!(2.0, 4.0, 5.0);
    assert_eq!(v!(29), &x.dot(y));

    let mut x = Value1d::new();
    assert!(x.is_empty());
    assert_eq!(x.len(), 0);
    assert_eq!(x.data(), vec![]);
    assert_eq!(x.sum().data(), 0.0);
    x.push(Value::from(1.0));
    assert_eq!(x.data(), vec![1.0]);
    assert_eq!(x.sum().data(), 1.0);
    assert!(!x.is_empty());
    assert_eq!(x.len(), 1);
    assert!(x.pop().unwrap().data() == 1.0);
    x.push(Value::from(2.0));
    x.clear();
    assert!(x.is_empty());
    assert_eq!(x.len(), 0);
    assert_eq!(x.data(), vec![]);
    x.push(Value::from(2.0));
    x.insert(0, Value::from(3.0));
    assert_eq!(x.data(), vec![3.0, 2.0]);
    assert_eq!(x.len(), 2);
    x.append(&mut v1d!(4.0, 5.0));
    assert_eq!(x.data(), vec![3.0, 2.0, 4.0, 5.0]);
    x.extend(v1d!(6.0, 7.0));
    assert_eq!(x.data(), vec![3.0, 2.0, 4.0, 5.0, 6.0, 7.0]);
}

#[test]
fn test_value2d() {
    let x = Value2d::zeros((2, 2));
    assert_eq!(x.shape(), (2, 2));
    assert_eq!(x, v2d![v1d!(0.0, 0.0), v1d!(0.0, 0.0)]);

    let mut x = Value2d::ones((2, 2));
    x.append(&mut Value2d::zeros((2, 2)));
    assert_eq!(x.shape(), (4, 2));
    assert_eq!(
        x,
        v2d![
            v1d!(1.0, 1.0),
            v1d!(1.0, 1.0),
            v1d!(0.0, 0.0),
            v1d!(0.0, 0.0)
        ]
    );
    x.extend(Value2d::ones((1, 2)).to_value1d());
    assert_eq!(x.shape(), (5, 2));
    assert_eq!(
        x,
        v2d![
            v1d!(1.0, 1.0),
            v1d!(1.0, 1.0),
            v1d!(0.0, 0.0),
            v1d!(0.0, 0.0),
            v1d!(1.0, 1.0)
        ]
    );

    let x = Value2d::ones((2, 2));
    assert_eq!(x.shape(), (2, 2));
    assert_eq!(x, v2d![v1d!(1.0, 1.0), v1d!(1.0, 1.0)]);

    let x = Value2d::from_value(Value::from(-1.0), (2, 2));
    assert_eq!(x.shape(), (2, 2));
    assert_eq!(x, v2d![v1d!(-1.0, -1.0), v1d!(-1.0, -1.0)]);

    assert_eq!(
        v2d![v1d!(1.0, 2.0), v1d!(3.0, 4.0)].pow(&Value2d::from_value(Value::from(2.0), (2, 2))),
        v2d![v1d!(1.0, 4.0), v1d!(9.0, 16.0)]
    );

    let ref x = vec![v1d!(1.0, 22.0, 33.0), v1d!(4.0, 5.0, 6.0)];
    let ref mx = Value2d::from(x.clone());
    assert!((x[0].variance().data() - 176.2222).abs() < 1e-4);
    assert!((x[0].variance_corr().data() - 264.3333).abs() < 1e-4);
    assert!((mx.variance().data() - 135.1389).abs() < 1e-4);
    assert!((mx.variance_corr().data() - 162.1667).abs() < 1e-4);
    assert!((x[0].std().data() - 13.2749).abs() < 1e-4);
    assert!((x[0].std_corr().data() - 16.2583).abs() < 1e-4);
    assert!((mx.std().data() - 11.6249).abs() < 1e-4);
    assert!((mx.std_corr().data() - 12.7345).abs() < 1e-4);

    let ref x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let ref y = vec![v1d!(7.0, 8.0, 9.0), v1d!(10.0, 11.0, 12.0)];

    let ref mx = Value2d::from(x.clone());
    let ref my = Value2d::from(y.clone());

    assert_eq!(mx.sum().data(), 21.0);
    assert_eq!(mx.mean().data(), 3.5);
    assert_eq!(mx.mse(&my).data(), 36.0);

    let actual = mx.softmax();
    let expected = v2d![
        v1d![0.00426978, 0.01160646, 0.03154963],
        v1d![0.08576079, 0.23312201, 0.63369132],
    ];
    let shape = actual.shape();
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            assert!((actual[(i, j)].data() - expected[(i, j)].data()).abs() < 1e-6);
        }
    }

    let actual = mx.softmax_axis_1();
    let expected = v2d![
        v1d![0.09003057, 0.24472847, 0.66524096],
        v1d![0.09003057, 0.24472847, 0.66524096],
    ];
    let shape = actual.shape();
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            assert!((actual[(i, j)].data() - expected[(i, j)].data()).abs() < 1e-6);
        }
    }

    let actual = mx.softmax_axis_0();
    let expected = v2d![
        v1d![0.04742587, 0.04742587, 0.04742587],
        v1d![0.95257413, 0.95257413, 0.95257413],
    ];
    let shape = actual.shape();
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            assert!((actual[(i, j)].data() - expected[(i, j)].data()).abs() < 1e-6);
        }
    }

    assert_eq!(mx.max().data(), 6.0);
    assert_eq!(mx.min().data(), 1.0);
    assert_eq!(my.shape(), (2, 3));
    assert_eq!(mx.shape(), (2, 3));
    assert_eq!(mx + my, v2d![v1d!(8.0, 10.0, 12.0), v1d!(14.0, 16.0, 18.0)]);
    assert_eq!(
        mx - my,
        v2d![v1d!(-6.0, -6.0, -6.0), v1d!(-6.0, -6.0, -6.0)]
    );
    assert_eq!(mx * my, v2d![v1d!(7.0, 16.0, 27.0), v1d!(40.0, 55.0, 72.0)]);
    let actual = mx / my;
    let expected = v2d![
        v1d!(1.0 / 7.0, 2.0 / 8.0, 3.0 / 9.0),
        v1d!(4.0 / 10.0, 5.0 / 11.0, 6.0 / 12.0),
    ];
    let shape = actual.shape();
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            assert!((actual[(i, j)].data() - expected[(i, j)].data()).abs() < 1e-6);
        }
    }

    let z = v!(3.0);
    assert_eq!(mx + z, v2d![v1d!(4.0, 5.0, 6.0), v1d!(7.0, 8.0, 9.0)]);
    assert_eq!(mx - z, v2d![v1d!(-2.0, -1.0, 0.0), v1d!(1.0, 2.0, 3.0)]);
    assert_eq!(mx * z, v2d![v1d!(3.0, 6.0, 9.0), v1d!(12.0, 15.0, 18.0)]);

    let actual = mx / z;
    let expected = v2d![
        v1d!(1.0 / 3.0, 2.0 / 3.0, 3.0 / 3.0),
        v1d!(4.0 / 3.0, 5.0 / 3.0, 6.0 / 3.0),
    ];
    let shape = actual.shape();
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            assert!((actual[(i, j)].data() - expected[(i, j)].data()).abs() < 1e-6);
        }
    }

    assert_eq!(z + mx, v2d![v1d!(4.0, 5.0, 6.0), v1d!(7.0, 8.0, 9.0)]);
    assert_eq!(z - mx, v2d![v1d!(2.0, 1.0, 0.0), v1d!(-1.0, -2.0, -3.0)]);
    assert_eq!(z * mx, v2d![v1d!(3.0, 6.0, 9.0), v1d!(12.0, 15.0, 18.0)]);
    let actual = z / mx;
    let expected = v2d![
        v1d!(3.0 / 1.0, 3.0 / 2.0, 3.0 / 3.0),
        v1d!(3.0 / 4.0, 3.0 / 5.0, 3.0 / 6.0),
    ];
    let shape = actual.shape();
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            assert!((actual[(i, j)].data() - expected[(i, j)].data()).abs() < 1e-6);
        }
    }

    let transposed = mx.transpose();
    assert_eq!(transposed.shape(), (3, 2));
    assert_eq!(
        transposed,
        v2d![v1d!(1.0, 4.0), v1d!(2.0, 5.0), v1d!(3.0, 6.0)]
    );

    let ref x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let ref y = vec![v1d!(7.0, 8.0), v1d!(9.0, 10.0), v1d!(11.0, 12.0)];

    let ref mx = Value2d::from(x.clone());
    let ref my = Value2d::from(y.clone());
    assert_eq!(mx.matmul(my), v2d![v1d!(58.0, 64.0), v1d!(139.0, 154.0)]);

    let ref x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let ref y = vec![v1d!(7.0, 8.0, 9.0), v1d!(10.0, 11.0, 12.0)];

    let ref mx = Value2d::from(x.clone());
    let ref my = Value2d::from(y.clone());
    assert_eq!(mx.to_value1d(), x.clone());
    assert_eq!(mx.sum_axis_0(), v1d![5.0, 7.0, 9.0]);
    assert_eq!(mx.sum_axis_1(), v1d![6.0, 15.0]);
    assert_eq!(mx.mean_axis_0(), v1d![2.5, 3.5, 4.5]);
    assert_eq!(mx.mean_axis_1(), v1d![2.0, 5.0]);
    assert_eq!(mx.max_axis_0(), v1d![4.0, 5.0, 6.0]);
    assert_eq!(mx.argmax_axis_0(), vec![(1, 0), (1, 1), (1, 2)]);
    assert_eq!(mx.max_axis_1(), v1d![3.0, 6.0]);
    assert_eq!(mx.argmax_axis_1(), vec![(0, 2), (1, 2)]);
    assert_eq!(mx.min_axis_0(), v1d![1.0, 2.0, 3.0]);
    assert_eq!(mx.argmin_axis_0(), vec![(0, 0), (0, 1), (0, 2)]);
    assert_eq!(mx.min_axis_1(), v1d![1.0, 4.0]);
    assert_eq!(mx.argmin_axis_1(), vec![(0, 0), (1, 0)]);
    assert_eq!(mx.mse_axis_0(&my), v1d![36.0, 36.0, 36.0]);
    assert_eq!(mx.mse_axis_1(&my), v1d![36.0, 36.0]);
    assert_eq!(mx[(0, 1)], *v!(2.0));
    assert_eq!(mx[(1, 0)], *v!(4.0));

    let ref x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let ref mx = Value2d::from(x.clone());
    let y: Value2d = mx
        .to_value1d()
        .iter()
        .map(|v| v.clone() + v!(1.0))
        .collect();
    let ref expected = vec![v1d!(2.0, 3.0, 4.0), v1d!(5.0, 6.0, 7.0)];
    let ref mexpected = Value2d::from(expected.clone());
    assert_eq!(y.data(), mexpected.data());
}

#[test]
fn test_iterators() {
    let x = v1d![1.0, 2.0];
    let mut expected = vec![1.0, 2.0];
    for i in &x {
        assert_eq!(i.data(), *expected.first().unwrap());
        expected.remove(0);
    }
    let mut expected = vec![1.0, 2.0];
    for i in x {
        assert_eq!(i.data(), *expected.first().unwrap());
        expected.remove(0);
    }

    let mx = v2d![v1d![1.0, 2.0], v1d![3.0, 4.0]];
    let mut expected = vec![1.0, 2.0, 3.0, 4.0];
    for i in &mx {
        assert_eq!(i.data(), *expected.first().unwrap());
        expected.remove(0);
    }
    let mut expected = vec![1.0, 2.0, 3.0, 4.0];
    for i in mx {
        assert_eq!(i.data(), *expected.first().unwrap());
        expected.remove(0);
    }
}

#[test]
#[should_panic = "different vector shapes"]
fn test_value2d_matmul_invalid_shape() {
    let ref x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let ref y = vec![v1d!(7.0, 8.0), v1d!(9.0, 10.0)];
    let ref mx = Value2d::from(x.clone());
    let ref my = Value2d::from(y.clone());
    let _ = mx.matmul(my);
}

#[test]
#[should_panic = "different vector shapes"]
fn test_vec_add_invalid_shape() {
    let ref x = v1d!(1.0, 2.0, 3.0);
    let ref y = v1d!(4.0, 5.0);
    let _ = x + y;
}

#[test]
#[should_panic = "different vector shapes"]
fn test_vec_sub_invalid_shape() {
    let ref x = v1d!(1.0, 2.0, 3.0);
    let ref y = v1d!(4.0, 5.0);
    let _ = x - y;
}

#[test]
#[should_panic = "different vector shapes"]
fn test_vec_mul_invalid_shape() {
    let ref x = v1d!(1.0, 2.0, 3.0);
    let ref y = v1d!(4.0, 5.0);
    let _ = x * y;
}

#[test]
#[should_panic = "different vector shapes"]
fn test_vec_div_invalid_shape() {
    let ref x = v1d!(1.0, 2.0, 3.0);
    let ref y = v1d!(4.0, 5.0);
    let _ = x / y;
}

#[test]
#[should_panic = "Division by zero"]
fn test_value2d_div_zero() {
    let ref x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let ref y = vec![v1d!(0.0, 0.0, 0.0), v1d!(0.0, 0.0, 0.0)];
    let ref mx = Value2d::from(x.clone());
    let ref my = Value2d::from(y.clone());
    let _ = mx / my;
}

#[test]
#[should_panic = "different Value2D shapes"]
fn test_value2d_add_invalid_shape() {
    let ref x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let ref y = vec![v1d!(0.0, 0.0), v1d!(0.0, 0.0)];
    let ref mx = Value2d::from(x.clone());
    let ref my = Value2d::from(y.clone());
    let _ = mx + my;
}

#[test]
#[should_panic = "different Value2D shapes"]
fn test_value2d_sub_invalid_shape() {
    let ref x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let ref y = vec![v1d!(0.0, 0.0), v1d!(0.0, 0.0)];
    let ref mx = Value2d::from(x.clone());
    let ref my = Value2d::from(y.clone());
    let _ = mx - my;
}

#[test]
#[should_panic = "different Value2D shapes"]
fn test_value2d_mul_invalid_shape() {
    let ref x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let ref y = vec![v1d!(0.0, 0.0), v1d!(0.0, 0.0)];
    let ref mx = Value2d::from(x.clone());
    let ref my = Value2d::from(y.clone());
    let _ = mx * my;
}

#[test]
#[should_panic = "different Value2D shapes"]
fn test_value2d_mse_invalid_shape() {
    let ref x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let ref y = vec![v1d!(0.0, 0.0), v1d!(0.0, 0.0)];
    let ref mx = Value2d::from(x.clone());
    let ref my = Value2d::from(y.clone());
    let _ = mx.mse(my);
}

#[test]
#[should_panic = "different Value2D shapes"]
fn test_value2d_mse_0_invalid_shape() {
    let ref x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let ref y = vec![v1d!(0.0, 0.0), v1d!(0.0, 0.0)];
    let ref mx = Value2d::from(x.clone());
    let ref my = Value2d::from(y.clone());
    let _ = mx.mse_axis_0(my);
}

#[test]
#[should_panic = "different Value2D shapes"]
fn test_value2d_mse_1_invalid_shape() {
    let ref x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let ref y = vec![v1d!(0.0, 0.0), v1d!(0.0, 0.0)];
    let ref mx = Value2d::from(x.clone());
    let ref my = Value2d::from(y.clone());
    let _ = mx.mse_axis_1(my);
}

#[test]
#[should_panic = "different Value2D shapes"]
fn test_value2d_pow_invalid_shape() {
    let ref x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let ref y = vec![v1d!(0.0, 0.0), v1d!(0.0, 0.0)];
    let ref mx = Value2d::from(x.clone());
    let ref my = Value2d::from(y.clone());
    let _ = mx.pow(my);
}

#[test]
#[should_panic = "different Value2D shapes"]
fn test_value2d_div_invalid_shape() {
    let ref x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let ref y = vec![v1d!(1.0, 1.0), v1d!(1.0, 1.0)];
    let ref mx = Value2d::from(x.clone());
    let ref my = Value2d::from(y.clone());
    let _ = mx / my;
}

#[test]
#[should_panic = "Division by zero"]
fn test_value2d_value_div_zero() {
    let ref x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let ref mx = Value2d::from(x.clone());
    let y = v!(0.0);
    let _ = mx / y;
}

#[test]
#[should_panic = "Division by zero"]
fn test_vec_div_zero() {
    let ref x = v1d!(1.0, 2.0, 3.0);
    let ref y = v1d!(0.0, 0.0, 0.0);
    let _ = x / y;
}

#[test]
#[should_panic = "Division by zero"]
fn test_vec_value_div_zero() {
    let ref x = v1d!(1.0, 2.0, 3.0);
    let y = v!(0.0);
    let _ = x / y;
}

#[test]
#[should_panic = "Division by zero"]
fn test_value_div_zero() {
    let x = v!(1.0);
    let y = v!(0.0);
    let _ = x / y;
}

#[test]
fn test_assign_ops() {
    let mut x = Value::from(2.0);
    let y = Value::from(4.0);
    x += &y;
    assert_eq!(x.data(), 6.0);
    x -= &y;
    assert_eq!(x.data(), 2.0);
    x *= &y;
    assert_eq!(x.data(), 8.0);
    x /= &y;
    assert_eq!(x.data(), 2.0);

    x += y.clone();
    assert_eq!(x.data(), 6.0);
    x -= y.clone();
    assert_eq!(x.data(), 2.0);
    x *= y.clone();
    assert_eq!(x.data(), 8.0);
    x /= y.clone();
    assert_eq!(x.data(), 2.0);

    let mut x = v1d!(1.0, 2.0, 3.0);
    let y = v1d!(4.0, 5.0, 6.0);
    let z = Value::from(2.0);
    x += &y;
    assert_eq!(x.data(), vec![5.0, 7.0, 9.0]);
    x -= &y;
    assert_eq!(x.data(), vec![1.0, 2.0, 3.0]);
    x *= &y;
    assert_eq!(x.data(), vec![4.0, 10.0, 18.0]);
    x /= &y;
    assert_eq!(x.data(), vec![1.0, 2.0, 3.0]);
    x += &z;
    assert_eq!(x.data(), vec![3.0, 4.0, 5.0]);
    x -= &z;
    assert_eq!(x.data(), vec![1.0, 2.0, 3.0]);
    x *= &z;
    assert_eq!(x.data(), vec![2.0, 4.0, 6.0]);
    x /= &z;
    assert_eq!(x.data(), vec![1.0, 2.0, 3.0]);

    x += y.clone();
    assert_eq!(x.data(), vec![5.0, 7.0, 9.0]);
    x -= y.clone();
    assert_eq!(x.data(), vec![1.0, 2.0, 3.0]);
    x *= y.clone();
    assert_eq!(x.data(), vec![4.0, 10.0, 18.0]);
    x /= y.clone();
    assert_eq!(x.data(), vec![1.0, 2.0, 3.0]);
    x += z.clone();
    assert_eq!(x.data(), vec![3.0, 4.0, 5.0]);
    x -= z.clone();
    assert_eq!(x.data(), vec![1.0, 2.0, 3.0]);
    x *= z.clone();
    assert_eq!(x.data(), vec![2.0, 4.0, 6.0]);
    x /= z.clone();
    assert_eq!(x.data(), vec![1.0, 2.0, 3.0]);

    let x = vec![v1d!(1.0, 2.0, 3.0), v1d!(4.0, 5.0, 6.0)];
    let mut mx = Value2d::from(x.clone());
    let y = vec![v1d!(7.0, 8.0, 9.0), v1d!(10.0, 11.0, 12.0)];
    let my = Value2d::from(y.clone());
    let z = Value::from(2.0);

    mx += &my;
    assert_eq!(
        mx.data(),
        vec![vec!(8.0, 10.0, 12.0), vec!(14.0, 16.0, 18.0)]
    );
    mx -= &my;
    assert_eq!(mx.data(), vec![vec!(1.0, 2.0, 3.0), vec!(4.0, 5.0, 6.0)]);
    mx *= &my;
    assert_eq!(
        mx.data(),
        vec![vec!(7.0, 16.0, 27.0), vec!(40.0, 55.0, 72.0)]
    );
    mx /= &my;
    assert_eq!(mx.data(), vec![vec!(1.0, 2.0, 3.0), vec!(4.0, 5.0, 6.0)]);

    mx += my.clone();
    assert_eq!(
        mx.data(),
        vec![vec!(8.0, 10.0, 12.0), vec!(14.0, 16.0, 18.0)]
    );
    mx -= my.clone();
    assert_eq!(mx.data(), vec![vec!(1.0, 2.0, 3.0), vec!(4.0, 5.0, 6.0)]);
    mx *= my.clone();
    assert_eq!(
        mx.data(),
        vec![vec!(7.0, 16.0, 27.0), vec!(40.0, 55.0, 72.0)]
    );
    mx /= my.clone();
    assert_eq!(mx.data(), vec![vec!(1.0, 2.0, 3.0), vec!(4.0, 5.0, 6.0)]);

    mx += &z;
    assert_eq!(mx.data(), vec![vec!(3.0, 4.0, 5.0), vec!(6.0, 7.0, 8.0)]);
    mx -= &z;
    assert_eq!(mx.data(), vec![vec!(1.0, 2.0, 3.0), vec!(4.0, 5.0, 6.0)]);
    mx *= &z;
    assert_eq!(mx.data(), vec![vec!(2.0, 4.0, 6.0), vec!(8.0, 10.0, 12.0)]);
    mx /= &z;
    assert_eq!(mx.data(), vec![vec!(1.0, 2.0, 3.0), vec!(4.0, 5.0, 6.0)]);

    mx += z.clone();
    assert_eq!(mx.data(), vec![vec!(3.0, 4.0, 5.0), vec!(6.0, 7.0, 8.0)]);
    mx -= z.clone();
    assert_eq!(mx.data(), vec![vec!(1.0, 2.0, 3.0), vec!(4.0, 5.0, 6.0)]);
    mx *= z.clone();
    assert_eq!(mx.data(), vec![vec!(2.0, 4.0, 6.0), vec!(8.0, 10.0, 12.0)]);
    mx /= z.clone();
    assert_eq!(mx.data(), vec![vec!(1.0, 2.0, 3.0), vec!(4.0, 5.0, 6.0)]);
}
