use crate::{Value, Value1d};

#[derive(Clone, Debug)]
pub struct Neuron {
    weights: Value1d,
    bias: Option<Value>,
    is_activation: bool,
}

impl Neuron {
    pub fn new(inputs: usize, is_activation: bool) -> Self {
        let weights = Value1d::rand(inputs);
        Self {
            weights,
            bias: Some(Value::from(0.0)),
            is_activation,
        }
    }

    pub fn without_bias(inputs: usize, is_activation: bool) -> Self {
        let weights = Value1d::rand(inputs);
        Self {
            weights,
            bias: None,
            is_activation,
        }
    }

    pub fn set_weights(&mut self, weights: &Value1d) {
        assert_eq!(weights.len(), self.weights.len());
        self.weights = weights.clone();
    }

    pub fn set_bias(&mut self, bias: &Value) {
        assert!(self.bias.is_some());
        self.bias = Some(bias.clone());
    }

    pub fn parameters(&self) -> Value1d {
        if let Some(bias) = &self.bias {
            return self
                .weights
                .iter()
                .chain(std::iter::once(bias))
                .cloned()
                .collect();
        }
        self.weights.clone()
    }

    pub fn forward(&self, inputs: &Value1d) -> Value {
        assert_eq!(inputs.len(), self.weights.len());
        let mut result = (inputs * &self.weights).sum();
        if let Some(bias) = &self.bias {
            result = result + bias;
        }
        if self.is_activation {
            return result.relu();
        }
        result
    }
}

#[derive(Clone, Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize, is_activation: bool) -> Self {
        let neurons = (0..outputs)
            .map(|_| Neuron::new(inputs, is_activation))
            .collect();
        Self { neurons }
    }

    pub fn without_bias(inputs: usize, outputs: usize, is_activation: bool) -> Self {
        let neurons = (0..outputs)
            .map(|_| Neuron::without_bias(inputs, is_activation))
            .collect();
        Self { neurons }
    }

    pub fn parameters(&self) -> Value1d {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }

    pub fn set_weights(&mut self, weight: Value) {
        for neuron in self.neurons.iter_mut() {
            let weights = Value1d::from(vec![weight.data(); neuron.weights.len()]);
            neuron.set_weights(&weights);
        }
    }

    pub fn set_bias(&mut self, bias: Value) {
        for neuron in self.neurons.iter_mut() {
            neuron.set_bias(&bias);
        }
    }

    pub fn forward(&self, inputs: &Value1d) -> Value1d {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(inputs: usize, outputs: Vec<usize>) -> Self {
        let mut dimensions = Vec::<usize>::new();
        dimensions.push(inputs);
        dimensions.extend(&outputs);
        let mut layers = Vec::<Layer>::new();
        for i in 0..outputs.len() {
            layers.push(Layer::new(
                dimensions[i],
                dimensions[i + 1],
                i != outputs.len() - 1,
            ));
        }
        Self { layers }
    }

    pub fn parameters(&self) -> Value1d {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    pub fn forward(&self, mut inputs: Value1d) -> Value1d {
        for layer in self.layers.iter() {
            inputs = layer.forward(&inputs);
        }
        inputs
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }
}
