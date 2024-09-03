use microgradr::{
    gbm::{Classifier, Regressor},
    v1d, Tree, Value, Value1d,
};

#[test]
fn test_tree() {
    let inputs = vec![
        v1d![1.0, 2.0, 3.0, 4.0],
        v1d![5.0, 3.0, 0.0, 0.0],
        v1d![4.0, 2.0, 9.0, 4.0],
        v1d![1.0, 1.0, 1.0, 8.0],
    ];
    let targets = v1d![3.0, 1.0, 4.0, 9.0];
    let model = Tree::new(inputs.clone(), targets.clone(), 1, 4);
    let predictions = inputs
        .into_iter()
        .map(|input| model.predict(input))
        .collect::<Value1d>();
    assert_eq!(0.0, predictions.mse(&targets).data());

    let test_inputs = vec![
        v1d![2.0, 2.0, 2.0, 2.0],
        v1d![1.0, 1.0, 1.0, 1.0],
        v1d![3.0, 3.0, 3.0, 3.0],
        v1d![4.0, 4.0, 4.0, 4.0],
    ];
    let test_targets = v1d![3.0, 9.0, 4.0, 4.0];
    let predictions = test_inputs
        .into_iter()
        .map(|input| model.predict(input))
        .collect::<Value1d>();
    assert_eq!(0.0, predictions.mse(&test_targets).data());
}

#[test]
fn test_gbm_regressor() {
    let inputs = vec![
        v1d![1.0, 2.0, 3.0, 4.0],
        v1d![5.0, 3.0, 0.0, 0.0],
        v1d![4.0, 2.0, 9.0, 4.0],
        v1d![1.0, 1.0, 1.0, 8.0],
    ];
    let targets = v1d![3.0, 1.0, 4.0, 9.0];
    let mut model = Regressor::new(10, 0.3, 1, 5);
    model.fit(inputs.clone(), targets.clone());
    let predictions = inputs
        .into_iter()
        .map(|input| model.predict(input))
        .collect::<Value1d>();
    assert!((0.006931 - predictions.mse(&targets).data()).abs() < 1e-6);

    let test_inputs = vec![
        v1d![2.0, 2.0, 2.0, 2.0],
        v1d![1.0, 1.0, 1.0, 1.0],
        v1d![3.0, 3.0, 3.0, 3.0],
        v1d![4.0, 4.0, 4.0, 4.0],
    ];
    let test_targets = v1d![3.0, 9.0, 4.0, 4.0];
    let predictions = test_inputs
        .into_iter()
        .map(|input| model.predict(input))
        .collect::<Value1d>();
    assert!((0.004837 - predictions.mse(&test_targets).data()).abs() < 1e-6);

    let inputs = vec![
        v1d![
            0.03807591,
            0.05068012,
            0.06169621,
            0.02187239,
            -0.0442235,
            -0.03482076,
            -0.04340085,
            -0.00259226,
            0.01990749,
            -0.01764613
        ],
        v1d![
            -0.00188202,
            -0.04464164,
            -0.05147406,
            -0.02632753,
            -0.00844872,
            -0.01916334,
            0.07441156,
            -0.03949338,
            -0.06833155,
            -0.09220405
        ],
        v1d![
            0.08529891,
            0.05068012,
            0.04445121,
            -0.00567042,
            -0.04559945,
            -0.03419447,
            -0.03235593,
            -0.00259226,
            0.00286131,
            -0.02593034
        ],
        v1d![
            -0.08906294,
            -0.04464164,
            -0.01159501,
            -0.03665608,
            0.01219057,
            0.02499059,
            -0.03603757,
            0.03430886,
            0.02268774,
            -0.00936191
        ],
        v1d![
            0.00538306,
            -0.04464164,
            -0.03638469,
            0.02187239,
            0.00393485,
            0.01559614,
            0.00814208,
            -0.00259226,
            -0.03198764,
            -0.04664087
        ],
        v1d![
            -0.09269548,
            -0.04464164,
            -0.04069594,
            -0.01944183,
            -0.06899065,
            -0.07928784,
            0.04127682,
            -0.0763945,
            -0.04117617,
            -0.09634616
        ],
        v1d![
            -0.04547248,
            0.05068012,
            -0.04716281,
            -0.01599898,
            -0.04009564,
            -0.02480001,
            0.00077881,
            -0.03949338,
            -0.06291688,
            -0.03835666
        ],
        v1d![
            0.06350368,
            0.05068012,
            -0.00189471,
            0.06662945,
            0.09061988,
            0.10891438,
            0.02286863,
            0.01770335,
            -0.03581619,
            0.00306441
        ],
        v1d![
            0.04170844,
            0.05068012,
            0.06169621,
            -0.04009893,
            -0.01395254,
            0.00620169,
            -0.02867429,
            -0.00259226,
            -0.01495969,
            0.01134862
        ],
        v1d![
            -0.07090025,
            -0.04464164,
            0.03906215,
            -0.03321323,
            -0.01257658,
            -0.03450761,
            -0.02499266,
            -0.00259226,
            0.06773705,
            -0.01350402
        ],
    ];
    let targets = v1d![151.0, 75.0, 141.0, 206.0, 135.0, 97.0, 138.0, 63.0, 110.0, 310.0];
    let mut model = Regressor::new(10, 0.3, 1, 5);
    model.fit(inputs.clone(), targets.clone());
    let predictions = inputs
        .into_iter()
        .map(|input| model.predict(input))
        .collect::<Value1d>();
    assert!((3.691966 - predictions.mse(&targets).data()).abs() < 1e-6);
}

#[test]
fn test_gbm_classifier() {
    let mut model = Classifier::new(2, 0.3, 1, 5);
    let (inputs, test_inputs, targets, test_targets) = gen_samples(1000, 4, 20);
    model.fit(inputs, targets);
    let predictions = test_inputs
        .into_iter()
        .map(|input| Value::from(model.predict(input)))
        .collect::<Value1d>();
    let test_targets = test_targets
        .into_iter()
        .map(|t| Value::from(t as f64))
        .collect::<Value1d>();
    assert!((3.5 - predictions.mse(&test_targets).data()).abs() < 1e-6);
}

fn gen_samples(
    n_samples: usize,
    n_classes: usize,
    n_features: usize,
) -> (Vec<Value1d>, Vec<Value1d>, Vec<usize>, Vec<usize>) {
    let mut inputs = Vec::with_capacity(n_samples);
    let mut targets = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let input = (0..n_features)
            .map(|f| Value::from(f as f64 * i as f64))
            .collect::<Value1d>();
        let target = i % n_classes;
        inputs.push(input);
        targets.push(target);
    }
    (
        inputs[..n_samples - 100].to_vec(),
        inputs[n_samples - 100..].to_vec(),
        targets[..n_samples - 100].to_vec(),
        targets[n_samples - 100..].to_vec(),
    )
}
