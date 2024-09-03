use crate::{Tree, Value, Value1d};

pub struct Regressor {
    learning_rate: Value,
    n_estimators: usize,
    min_samples_leaf: usize,
    max_depth: usize,
    trees: Vec<Tree>,
    f0: Value,
}

impl Regressor {
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        min_samples_leaf: usize,
        max_depth: usize,
    ) -> Self {
        Self {
            learning_rate: Value::from(learning_rate),
            n_estimators,
            min_samples_leaf,
            max_depth,
            trees: Vec::new(),
            f0: Value::from(0.0),
        }
    }

    pub fn fit(&mut self, inputs: Vec<Value1d>, targets: Value1d) -> () {
        assert!(!inputs.is_empty());
        assert_eq!(inputs.len(), targets.len());
        self.f0 = targets.clone().into_iter().collect::<Value1d>().mean();
        let v = Value1d::from(vec![self.f0.data()]);
        let mut fm = vec![v; targets.len()];
        for _ in 0..self.n_estimators {
            let residuals = targets
                .clone()
                .into_iter()
                .zip(fm.clone())
                .map(|(t, fm)| t - &fm)
                .collect();
            let tree = Tree::new(
                inputs.clone(),
                residuals,
                self.min_samples_leaf,
                self.max_depth,
            );
            let predictions = inputs
                .clone()
                .into_iter()
                .map(|input| tree.predict(input))
                .collect::<Value1d>();
            fm = predictions
                .clone()
                .into_iter()
                .zip(fm)
                .map(|(p, f)| f + &self.learning_rate * p)
                .collect();
            self.trees.push(tree);
        }
    }

    pub fn predict(&self, inputs: Value1d) -> Value {
        let mut prediction = self.f0.clone();
        for tree in &self.trees {
            prediction = prediction + &self.learning_rate * tree.predict(inputs.clone());
        }
        prediction
    }
}

pub struct Classifier {
    learning_rate: Value,
    n_estimators: usize,
    min_samples_leaf: usize,
    max_depth: usize,
    boosters: Vec<Vec<Tree>>,
}

impl Classifier {
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        min_samples_leaf: usize,
        max_depth: usize,
    ) -> Self {
        Self {
            learning_rate: Value::from(learning_rate),
            n_estimators,
            min_samples_leaf,
            max_depth,
            boosters: Vec::new(),
        }
    }

    pub fn fit(&mut self, inputs: Vec<Value1d>, targets: Vec<usize>) -> () {
        assert!(!inputs.is_empty());
        assert_eq!(inputs.len(), targets.len());
        let (targets_encoded, classes) = Self::one_hot_encode(targets);
        let mut predictions = targets_encoded
            .iter()
            .map(|v| Value1d::from(vec![0.0; v.len()]))
            .collect::<Vec<Value1d>>();
        let mut probabilities = Self::softmax(predictions.clone());
        self.boosters.clear();
        for _ in 0..self.n_estimators {
            let mut trees = Vec::new();
            for idx in 0..classes {
                let negative_gradients =
                    Self::negative_gradients(targets_encoded.clone(), probabilities.clone(), idx);
                let hessians = Self::hessians(probabilities.clone(), idx);
                let mut tree = Tree::new(
                    inputs.clone(),
                    negative_gradients.clone(),
                    self.min_samples_leaf,
                    self.max_depth,
                );
                Self::update_leaves(
                    &mut tree,
                    inputs.clone(),
                    negative_gradients.clone(),
                    hessians,
                );
                for i in 0..predictions.len() {
                    predictions[idx] =
                        &predictions[idx] + &self.learning_rate * tree.predict(inputs[i].clone());
                }
                probabilities = Self::softmax(predictions.clone());
                trees.push(tree);
            }
            self.boosters.push(trees);
        }
    }

    fn update_leaves(
        tree: &mut Tree,
        inputs: Vec<Value1d>,
        negative_gradients: Value1d,
        hessians: Value1d,
    ) -> () {
        let leaf_node_for_inputs = inputs
            .iter()
            .map(|input| tree.apply(input.clone()))
            .collect::<Vec<usize>>();
        for leaf in 0..tree.leaves() {
            let inputs_in_leaf = leaf_node_for_inputs
                .clone()
                .into_iter()
                .enumerate()
                .filter(|(_, i)| *i == leaf)
                .map(|(i, _)| i)
                .collect::<Vec<usize>>();
            let negative_gradients_in_leaf = inputs_in_leaf
                .clone()
                .into_iter()
                .map(|idx| negative_gradients[idx].clone())
                .collect::<Value1d>();
            let hessians_in_leaf = inputs_in_leaf
                .into_iter()
                .map(|idx| hessians[idx].clone())
                .collect::<Value1d>();
            let value = negative_gradients_in_leaf.sum() / hessians_in_leaf.sum();
            tree.update_leaf(leaf, value);
        }
    }

    fn negative_gradients(targets: Vec<Value1d>, predictions: Vec<Value1d>, idx: usize) -> Value1d {
        targets
            .into_iter()
            .zip(predictions)
            .map(|(t, p)| &t[idx] - &p[idx])
            .collect()
    }

    fn hessians(predictions: Vec<Value1d>, idx: usize) -> Value1d {
        predictions
            .into_iter()
            .map(|p| &p[idx] * (Value::from(1.0) - &p[idx]))
            .collect()
    }

    fn softmax(predictions: Vec<Value1d>) -> Vec<Value1d> {
        predictions.into_iter().map(|v| v.softmax()).collect()
    }

    fn one_hot_encode(values: Vec<usize>) -> (Vec<Value1d>, usize) {
        assert!(!values.is_empty());
        let classes = values.iter().max().unwrap() + 1;
        (
            values
                .into_iter()
                .map(|value| {
                    let mut v = vec![0.0; classes];
                    v[value] = 1.0;
                    Value1d::from(v)
                })
                .collect(),
            classes,
        )
    }

    fn predict_proba(&self, inputs: Value1d) -> Value1d {
        let classes = self.boosters[0].len();
        let mut probabilities = Value1d::from(vec![0.0; classes]);
        for trees in &self.boosters {
            for (idx, tree) in trees.iter().enumerate() {
                probabilities[idx] =
                    &probabilities[idx] + &self.learning_rate * tree.predict(inputs.clone());
            }
        }
        probabilities.softmax()
    }

    pub fn predict(&self, inputs: Value1d) -> usize {
        self.predict_proba(inputs).argmax()
    }
}

pub struct BinaryClassifier {
    classifier: Classifier,
}

impl BinaryClassifier {
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        min_samples_leaf: usize,
        max_depth: usize,
    ) -> Self {
        Self {
            classifier: Classifier::new(n_estimators, learning_rate, min_samples_leaf, max_depth),
        }
    }

    pub fn fit(&mut self, inputs: Vec<Value1d>, targets: Vec<bool>) -> () {
        let targets_as_values = targets.iter().map(|&x| x as usize).collect();
        self.classifier.fit(inputs, targets_as_values);
    }

    pub fn predict(&self, inputs: Value1d) -> bool {
        self.classifier.predict(inputs) != 0
    }
}
