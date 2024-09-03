use std::{cell::RefCell, rc::Rc};

use crate::{Value, Value1d};

struct Node {
    value: Value,
    left: Option<NodeRef>,
    right: Option<NodeRef>,
    min_samples_leaf: usize,
    max_depth: usize,
    feature_idx: usize,
    score: Value,
    threshold: Value,
    leaf_id: usize,
}

type NodeRef = Rc<RefCell<Node>>;

impl Node {
    pub fn new(
        inputs: &Vec<Value1d>,
        targets: &Value1d,
        min_samples_leaf: usize,
        max_depth: usize,
        indices: &Vec<usize>,
    ) -> Self {
        assert!(inputs.len() > 0);
        assert!(inputs.len() == targets.len());
        assert!(min_samples_leaf > 0);
        let mut indices = indices.clone();
        if indices.len() == 0 {
            indices = (0..targets.len()).collect();
        }
        let value = indices
            .iter()
            .map(|i| targets[*i].clone())
            .collect::<Value1d>()
            .mean();
        let mut node = Node {
            value,
            feature_idx: 0,
            threshold: Value::from(f64::INFINITY),
            score: Value::from(f64::INFINITY),
            left: None,
            right: None,
            min_samples_leaf,
            max_depth,
            leaf_id: 0,
        };

        if node.max_depth > 0 {
            node.add_children(&indices, inputs, targets);
        }
        node
    }

    fn is_leaf(&self) -> bool {
        self.score.data() == f64::INFINITY
    }

    fn add_children(
        &mut self,
        indices: &Vec<usize>,
        inputs: &Vec<Value1d>,
        targets: &Value1d,
    ) -> () {
        let features = inputs[0].len();
        for i in 0..features {
            self.find_split(indices, inputs, targets, i);
        }
        if self.is_leaf() {
            return;
        }
        let left_indices: Vec<usize> = indices
            .clone()
            .into_iter()
            .filter(|i| inputs[*i][self.feature_idx] <= self.threshold)
            .collect();
        let right_indices: Vec<usize> = indices
            .clone()
            .into_iter()
            .filter(|i| inputs[*i][self.feature_idx] > self.threshold)
            .collect();
        self.left = Some(Rc::new(RefCell::new(Node::new(
            inputs,
            targets,
            self.min_samples_leaf,
            self.max_depth - 1,
            &left_indices,
        ))));
        self.right = Some(Rc::new(RefCell::new(Node::new(
            inputs,
            targets,
            self.min_samples_leaf,
            self.max_depth - 1,
            &right_indices,
        ))));
    }

    fn find_split(
        &mut self,
        indices: &Vec<usize>,
        inputs: &Vec<Value1d>,
        targets: &Value1d,
        feature_idx: usize,
    ) -> () {
        let samples = indices.len();
        let mut sorted_indices = indices.clone();
        sorted_indices.sort_by(|a, b| {
            inputs[*a][feature_idx]
                .partial_cmp(&inputs[*b][feature_idx])
                .unwrap()
        });
        let sorted_inputs = sorted_indices
            .iter()
            .map(|i| inputs[*i][feature_idx].clone())
            .collect::<Value1d>();
        let sorted_targets = sorted_indices
            .iter()
            .map(|i| targets[*i].clone())
            .collect::<Value1d>();
        let (sum_targets, len_targets) =
            (sorted_targets.iter().sum::<Value>(), sorted_targets.len());
        let (mut sum_right, mut len_right) = (sum_targets.clone(), len_targets);
        let (mut sum_left, mut len_left) = (Value::from(0.0), 0 as usize);
        for i in 0..(samples - self.min_samples_leaf) {
            let (target_i, input_i, input_i_next) =
                (&sorted_targets[i], &sorted_inputs[i], &sorted_inputs[i + 1]);
            sum_left = sum_left + target_i;
            sum_right = sum_right - target_i;
            len_left += 1;
            len_right -= 1;
            if len_left < self.min_samples_leaf || input_i == input_i_next {
                continue;
            }
            let score = -sum_left.pow(&Value::from(2.0)) / Value::from(len_left)
                - sum_right.pow(&Value::from(2.0)) / Value::from(len_right)
                + sum_targets.pow(&Value::from(2.0)) / Value::from(len_targets);
            if score < self.score {
                self.score = score;
                self.feature_idx = feature_idx;
                self.threshold = (input_i + input_i_next) / Value::from(2.0);
            }
        }
    }

    fn predict(&self, inputs: &Value1d) -> (Value, usize) {
        if self.is_leaf() {
            return (self.value.clone(), self.leaf_id);
        }
        if inputs[self.feature_idx] <= self.threshold {
            return self.left.as_ref().unwrap().borrow().predict(inputs);
        }
        self.right.as_ref().unwrap().borrow().predict(inputs)
    }
}

pub struct Tree {
    root: NodeRef,
    leaves: Vec<NodeRef>,
}

impl Tree {
    pub fn new(
        inputs: Vec<Value1d>,
        targets: Value1d,
        min_samples_leaf: usize,
        max_depth: usize,
    ) -> Self {
        let root = Rc::new(RefCell::new(Node::new(
            &inputs,
            &targets,
            min_samples_leaf,
            max_depth,
            &Vec::new(),
        )));
        let leaves = Self::leaves_vec(root.clone());
        Tree { root, leaves }
    }

    fn leaves_vec(root: NodeRef) -> Vec<NodeRef> {
        let mut leaves: Vec<NodeRef> = Vec::new();
        let mut stack = vec![root.clone()];
        while !stack.is_empty() {
            let node = stack.pop().unwrap();
            if node.borrow().is_leaf() {
                node.borrow_mut().leaf_id = leaves.len();
                leaves.push(node.clone());
            }
            if node.borrow().left.is_some() {
                stack.push(node.borrow().left.as_ref().unwrap().clone());
            }
            if node.borrow().right.is_some() {
                stack.push(node.borrow().right.as_ref().unwrap().clone());
            }
        }
        leaves
    }

    pub fn leaves(&self) -> usize {
        self.leaves.len()
    }

    pub fn update_leaf(&mut self, leaf: usize, value: Value) -> () {
        assert!(leaf < self.leaves.len());
        self.leaves[leaf].borrow_mut().value = value;
    }

    pub fn predict(&self, inputs: Value1d) -> Value {
        self.root.borrow().predict(&inputs).0
    }

    pub fn apply(&self, inputs: Value1d) -> usize {
        self.root.borrow().predict(&inputs).1
    }
}
