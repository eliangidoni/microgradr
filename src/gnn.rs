use crate::{Value, Value1d, Value2d};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum ArgType {
    Value(Value),
    Value1d(Value1d),
    Value2d(Value2d),
}

pub trait MessagePassingTrait {
    fn aggregate(&self, aggregated: &mut Value2d, target_nodes: &[usize], messages: &Value2d) {
        // sum aggregation
        let feature_dim = messages.shape().1;
        for (edge_idx, &target_node) in target_nodes.iter().enumerate() {
            for feat_idx in 0..feature_dim {
                aggregated[(target_node, feat_idx)] += messages[(edge_idx, feat_idx)].clone();
            }
        }
    }

    /// returns message matrix with shape [E, feature_dim]
    /// x_j: source node features for all edges [E, feature_dim]
    /// x_i: target node features for all edges [E, feature_dim]
    fn message(
        &self,
        x_j: &Value2d,
        _x_i: &Value2d,
        _args: Option<&HashMap<String, ArgType>>,
    ) -> Value2d {
        x_j.clone()
    }

    /// aggr_out: aggregated features from neighbors
    /// x: original node features
    fn update_node_embeddings(
        &self,
        aggr_out: &Value2d,
        _x: &Value2d,
        _args: Option<&HashMap<String, ArgType>>,
    ) -> Value2d {
        aggr_out.clone()
    }

    /// edge_index: graph connectivity as (2, num_edges) matrix
    /// x: node features (num_nodes, feature_dim)
    /// args: optional additional arguments passed to message/update functions
    /// Returns aggregated features for each node
    fn propagate(
        &self,
        edge_index: &Value2d,
        x: &Value2d,
        args: Option<&HashMap<String, ArgType>>,
    ) -> Value2d {
        assert_eq!(
            edge_index.shape().0,
            2,
            "edge_index must have shape (2, num_edges)"
        );

        let num_nodes = x.shape().0;
        let feature_dim = x.shape().1;
        let num_edges = edge_index.shape().1;
        let mut aggregated = Value2d::zeros((num_nodes, feature_dim));
        let mut source_features = Value2d::zeros((num_edges, feature_dim));
        let mut target_features = Value2d::zeros((num_edges, feature_dim));
        let mut target_nodes = Vec::with_capacity(num_edges);
        for edge_idx in 0..num_edges {
            let source_node = edge_index[(0, edge_idx)].data() as usize;
            let target_node = edge_index[(1, edge_idx)].data() as usize;

            target_nodes.push(target_node);

            for feat_idx in 0..feature_dim {
                source_features[(edge_idx, feat_idx)] = x[(source_node, feat_idx)].clone();
                target_features[(edge_idx, feat_idx)] = x[(target_node, feat_idx)].clone();
            }
        }
        let messages = self.message(&source_features, &target_features, args);
        self.aggregate(&mut aggregated, &target_nodes, &messages);
        self.update_node_embeddings(&aggregated, x, args)
    }

    /// Returns new edge_index with remaining self-loops added
    fn add_remaining_self_loops(&self, edge_index: &Value2d, num_nodes: usize) -> Value2d {
        let num_edges = edge_index.shape().1;

        // Find which nodes already have self-loops
        let mut has_self_loop = vec![false; num_nodes];
        for edge_idx in 0..num_edges {
            let source_node = edge_index[(0, edge_idx)].data() as usize;
            let target_node = edge_index[(1, edge_idx)].data() as usize;
            if source_node == target_node {
                has_self_loop[source_node] = true;
            }
        }

        // Count how many self-loops we need to add
        let missing_self_loops = has_self_loop.iter().filter(|&&has_loop| !has_loop).count();
        let total_edges = num_edges + missing_self_loops;

        let mut result = Value2d::zeros((2, total_edges));

        // Copy existing edges
        for edge_idx in 0..num_edges {
            result[(0, edge_idx)] = edge_index[(0, edge_idx)].clone();
            result[(1, edge_idx)] = edge_index[(1, edge_idx)].clone();
        }

        // Add missing self-loops
        let mut new_edge_idx = num_edges;
        for node_idx in 0..num_nodes {
            if !has_self_loop[node_idx] {
                result[(0, new_edge_idx)] = Value::from(node_idx as f64);
                result[(1, new_edge_idx)] = Value::from(node_idx as f64);
                new_edge_idx += 1;
            }
        }

        result
    }
}

pub struct MessagePassing;

impl MessagePassingTrait for MessagePassing {}

impl MessagePassing {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, edge_index: &Value2d, x: &Value2d) -> Value2d {
        self.propagate(edge_index, x, None)
    }
}

pub struct GCNConvLayer {
    weight: Value2d,
    bias: Option<Value1d>,
}

impl MessagePassingTrait for GCNConvLayer {
    fn message(
        &self,
        x_j: &Value2d,
        _x_i: &Value2d,
        args: Option<&HashMap<String, ArgType>>,
    ) -> Value2d {
        assert!(args.is_some(), "GCN message function requires args");
        assert!(
            args.as_ref().unwrap().contains_key("edge_norms"),
            "GCN message function requires 'edge_norms' in args"
        );
        let edge_norms = match args.as_ref().unwrap().get("edge_norms").unwrap() {
            ArgType::Value1d(v) => v,
            _ => panic!("Expected edge_norms to be Value1d"),
        };
        let num_edges = x_j.shape().0;
        let feature_dim = x_j.shape().1;
        let mut normalized = Value2d::zeros((num_edges, feature_dim));
        for edge_idx in 0..num_edges {
            assert!(
                edge_idx < edge_norms.len(),
                "Edge index out of bounds for edge_norms"
            );
            let norm = &edge_norms[edge_idx];
            for feat_idx in 0..feature_dim {
                normalized[(edge_idx, feat_idx)] = x_j[(edge_idx, feat_idx)].clone() * norm.clone();
            }
        }
        normalized
    }
}

impl GCNConvLayer {
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let weight = Value2d::rand((in_features, out_features));
        let bias = if use_bias {
            Some(Value1d::rand(out_features))
        } else {
            None
        };

        Self { weight, bias }
    }

    pub fn parameters(&self) -> Value1d {
        if let Some(bias) = &self.bias {
            return self.weight.iter().chain(bias.iter()).cloned().collect();
        }
        self.weight.iter().cloned().collect()
    }

    pub fn zero_grad(&mut self) {
        self.parameters().zero_grad();
    }

    pub fn update(&mut self, learning_rate: f64) {
        self.parameters().update(learning_rate);
    }

    pub fn set_weights(&mut self, weight: Value) {
        self.weight = Value2d::ones(self.weight.shape()) * weight;
    }

    pub fn set_bias(&mut self, bias: Value) {
        assert!(
            self.bias.is_some(),
            "This GCN layer was created without bias"
        );
        self.bias = Some(Value1d::ones(self.bias.as_ref().unwrap().len()) * bias);
    }

    /// edge_index: graph connectivity as (2, num_edges) matrix
    /// x: node features (num_nodes, in_features)
    /// Returns: updated node features (num_nodes, out_features)
    pub fn forward(&self, edge_index: &Value2d, x: &Value2d) -> Value2d {
        let transformed = x.matmul(&self.weight);

        let num_nodes = transformed.shape().0;
        let edge_index_with_self_loops = self.add_remaining_self_loops(edge_index, num_nodes);

        // only count incoming edges (col indices)
        let num_edges = edge_index_with_self_loops.shape().1;
        let mut degrees = vec![0.0_f64; num_nodes];

        for edge_idx in 0..num_edges {
            let target_node = edge_index_with_self_loops[(1, edge_idx)].data() as usize;
            degrees[target_node] += 1.0;
        }

        let mut edge_norms = Vec::new();
        for edge_idx in 0..num_edges {
            let source_node = edge_index_with_self_loops[(0, edge_idx)].data() as usize;
            let target_node = edge_index_with_self_loops[(1, edge_idx)].data() as usize;

            let source_deg_inv_sqrt = if degrees[source_node] > 0.0 {
                1.0 / degrees[source_node].sqrt()
            } else {
                0.0
            };
            let target_deg_inv_sqrt = if degrees[target_node] > 0.0 {
                1.0 / degrees[target_node].sqrt()
            } else {
                0.0
            };
            edge_norms.push(source_deg_inv_sqrt * target_deg_inv_sqrt);
        }

        let mut args = HashMap::new();
        args.insert(
            "edge_norms".to_string(),
            ArgType::Value1d(Value1d::from(edge_norms)),
        );
        let aggregated = self.propagate(&edge_index_with_self_loops, &transformed, Some(&args));
        let output = if let Some(ref bias) = self.bias {
            aggregated + Value2d::from(vec![bias.clone(); x.shape().0])
        } else {
            aggregated
        };
        output
    }
}

pub struct SAGEConvLayer {
    root_weight: Value2d,
    weight: Value2d,
    bias: Option<Value1d>,
}

impl MessagePassingTrait for SAGEConvLayer {
    fn aggregate(&self, aggregated: &mut Value2d, target_nodes: &[usize], messages: &Value2d) {
        // mean aggregation
        let feature_dim = messages.shape().1;
        let num_nodes = aggregated.shape().0;
        let mut node_counts = vec![0; num_nodes];
        for &target_node in target_nodes.iter() {
            node_counts[target_node] += 1;
        }
        for (edge_idx, &target_node) in target_nodes.iter().enumerate() {
            for feat_idx in 0..feature_dim {
                aggregated[(target_node, feat_idx)] += messages[(edge_idx, feat_idx)].clone();
            }
        }
        for node_idx in 0..num_nodes {
            if node_counts[node_idx] > 0 {
                let count = Value::from(node_counts[node_idx] as f64);
                for feat_idx in 0..feature_dim {
                    aggregated[(node_idx, feat_idx)] /= count.clone();
                }
            }
        }
    }
}

impl SAGEConvLayer {
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let root_weight = Value2d::rand((in_features, out_features));
        let neighbor_weight = Value2d::rand((in_features, out_features));
        let bias = if use_bias {
            Some(Value1d::rand(out_features))
        } else {
            None
        };

        Self {
            root_weight: root_weight,
            weight: neighbor_weight,
            bias,
        }
    }

    pub fn parameters(&self) -> Value1d {
        if let Some(bias) = &self.bias {
            return self
                .root_weight
                .iter()
                .chain(self.weight.iter())
                .chain(bias.iter())
                .cloned()
                .collect();
        }
        self.root_weight
            .iter()
            .chain(self.weight.iter())
            .cloned()
            .collect()
    }

    pub fn zero_grad(&mut self) {
        self.parameters().zero_grad();
    }

    pub fn update(&mut self, learning_rate: f64) {
        self.parameters().update(learning_rate);
    }

    pub fn set_root_weights(&mut self, weight: Value) {
        self.root_weight = Value2d::ones(self.root_weight.shape()) * weight;
    }

    pub fn set_weights(&mut self, weight: Value) {
        self.weight = Value2d::ones(self.weight.shape()) * weight;
    }

    pub fn set_bias(&mut self, bias: Value) {
        assert!(
            self.bias.is_some(),
            "This SAGE layer was created without bias"
        );
        self.bias = Some(Value1d::ones(self.bias.as_ref().unwrap().len()) * bias);
    }

    /// edge_index: graph connectivity as (2, num_edges) matrix
    /// x: node features (num_nodes, in_features)
    /// Returns: updated node features (num_nodes, out_features)
    pub fn forward(&self, edge_index: &Value2d, x: &Value2d) -> Value2d {
        let root_features = x.matmul(&self.root_weight);
        let neighbor_aggregated = self.propagate(edge_index, x, None);
        let neighbor_features = neighbor_aggregated.matmul(&self.weight);
        let combined = root_features + neighbor_features;
        let output = if let Some(ref bias) = self.bias {
            combined + Value2d::from(vec![bias.clone(); x.shape().0])
        } else {
            combined
        };
        output
    }
}

#[derive(Clone, Debug)]
pub struct DropoutLayer {
    is_training: bool,
    rate: f64,
}

impl DropoutLayer {
    pub fn new(rate: f64) -> Self {
        assert!(rate >= 0.0 && rate <= 1.0);
        Self {
            is_training: true,
            rate,
        }
    }

    pub fn forward(&self, input: &Value2d) -> Value2d {
        if !self.is_training {
            return input.clone();
        }
        let mut output = Value2d::zeros(input.shape());
        for i in 0..input.shape().0 {
            for j in 0..input.shape().1 {
                if crate::gen_range(0.0, 1.0) >= self.rate {
                    output[(i, j)] = input[(i, j)].clone();
                }
            }
        }
        output
    }

    pub fn train(&mut self) {
        self.is_training = true;
    }

    pub fn eval(&mut self) {
        self.is_training = false;
    }
}

pub struct GCN {
    layers: Vec<GCNConvLayer>,
    dropout: Option<DropoutLayer>,
}

impl GCN {
    pub fn new(
        in_features: usize,
        out_features: usize,
        hidden_features: usize,
        dropout_rate: f64,
        num_layers: usize,
    ) -> Self {
        assert!(num_layers > 0, "Number of layers must be > 0");
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let isize = if i == 0 { in_features } else { hidden_features };
            layers.push(GCNConvLayer::new(isize, hidden_features, true));
        }
        let final_layer = GCNConvLayer::new(hidden_features, out_features, true);
        layers.push(final_layer);
        let dropout = if dropout_rate > 0.0 {
            Some(DropoutLayer::new(dropout_rate))
        } else {
            None
        };
        Self { layers, dropout }
    }

    pub fn eval(&mut self) {
        if let Some(ref mut dropout) = self.dropout {
            dropout.eval();
        }
    }

    pub fn train(&mut self) {
        if let Some(ref mut dropout) = self.dropout {
            dropout.train();
        }
    }

    pub fn set_bias(&mut self, bias: Value) {
        for layer in &mut self.layers {
            layer.set_bias(bias.clone());
        }
    }

    pub fn set_weights(&mut self, weight: Value) {
        for layer in &mut self.layers {
            layer.set_weights(weight.clone());
        }
    }

    pub fn forward(&self, edge_index: &Value2d, x: &Value2d) -> Value2d {
        let mut output = x.clone();
        for i in 0..self.layers.len() - 1 {
            output = self.layers[i].forward(edge_index, &output).relu();
            output = if let Some(ref dropout) = self.dropout {
                dropout.forward(&output)
            } else {
                output
            };
        }
        self.layers.last().unwrap().forward(edge_index, &output)
    }

    pub fn parameters(&self) -> Value1d {
        self.layers
            .iter()
            .map(|layer| layer.parameters())
            .flatten()
            .collect()
    }

    pub fn zero_grad(&mut self) {
        self.parameters().zero_grad();
    }

    pub fn update(&mut self, learning_rate: f64) {
        self.parameters().update(learning_rate);
    }
}

pub struct SAGE {
    layers: Vec<SAGEConvLayer>,
    dropout: Option<DropoutLayer>,
}

impl SAGE {
    pub fn new(
        in_features: usize,
        out_features: usize,
        hidden_features: usize,
        dropout_rate: f64,
        num_layers: usize,
    ) -> Self {
        assert!(num_layers > 0, "Number of layers must be > 0");
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let isize = if i == 0 { in_features } else { hidden_features };
            layers.push(SAGEConvLayer::new(isize, hidden_features, true));
        }
        let final_layer = SAGEConvLayer::new(hidden_features, out_features, true);
        layers.push(final_layer);

        let dropout = if dropout_rate > 0.0 {
            Some(DropoutLayer::new(dropout_rate))
        } else {
            None
        };
        Self { layers, dropout }
    }

    pub fn eval(&mut self) {
        if let Some(ref mut dropout) = self.dropout {
            dropout.eval();
        }
    }

    pub fn train(&mut self) {
        if let Some(ref mut dropout) = self.dropout {
            dropout.train();
        }
    }

    pub fn set_root_weights(&mut self, weight: Value) {
        for layer in &mut self.layers {
            layer.set_root_weights(weight.clone());
        }
    }

    pub fn set_neighbor_weights(&mut self, weight: Value) {
        for layer in &mut self.layers {
            layer.set_weights(weight.clone());
        }
    }

    pub fn set_bias(&mut self, bias: Value) {
        for layer in &mut self.layers {
            layer.set_bias(bias.clone());
        }
    }

    pub fn forward(&self, edge_index: &Value2d, x: &Value2d) -> Value2d {
        let mut output = x.clone();
        for i in 0..self.layers.len() - 1 {
            output = self.layers[i].forward(edge_index, &output).relu();
            output = if let Some(ref dropout) = self.dropout {
                dropout.forward(&output)
            } else {
                output
            };
        }
        self.layers.last().unwrap().forward(edge_index, &output)
    }

    pub fn parameters(&self) -> Value1d {
        self.layers
            .iter()
            .map(|layer| layer.parameters())
            .flatten()
            .collect()
    }

    pub fn zero_grad(&mut self) {
        self.parameters().zero_grad();
    }

    pub fn update(&mut self, learning_rate: f64) {
        self.parameters().update(learning_rate);
    }
}
