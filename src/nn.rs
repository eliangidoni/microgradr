use std::cell::RefCell;

use rand::{distributions::uniform::SampleUniform, thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{Value, ValueVec};

thread_local! {
    static RNG: RefCell<ChaCha20Rng> = RefCell::new(ChaCha20Rng::from_rng(thread_rng()).unwrap());
}

pub fn set_seed(seed: u64) -> () {
    RNG.set(ChaCha20Rng::seed_from_u64(seed));
}

pub fn gen_range<T>(low: T, high: T) -> T
where
    T: SampleUniform + PartialOrd,
{
    RNG.with_borrow_mut(|rng| rng.gen_range(low..high))
}

pub struct Neuron {
    weights: ValueVec,
    bias: Value,
    is_activation: bool,
}

impl Neuron {
    pub fn new(inputs: usize, is_activation: bool) -> Self {
        let rands = (0..inputs)
            .map(|_| gen_range(-1.0, 1.0))
            .collect::<Vec<f64>>();
        let weights = ValueVec::from(rands);
        Self {
            weights,
            bias: Value::from(0.0),
            is_activation,
        }
    }

    pub fn parameters(&self) -> ValueVec {
        self.weights
            .iter()
            .chain(std::iter::once(&self.bias))
            .cloned()
            .collect()
    }

    pub fn forward(&self, inputs: &ValueVec) -> Value {
        assert_eq!(inputs.len(), self.weights.len());
        let result = (inputs * &self.weights).sum() + &self.bias;
        if self.is_activation {
            return result.relu();
        }
        result
    }
}

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

    pub fn parameters(&self) -> ValueVec {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }

    pub fn forward(&self, inputs: &ValueVec) -> ValueVec {
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

    pub fn parameters(&self) -> ValueVec {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    pub fn forward(&self, mut inputs: ValueVec) -> ValueVec {
        for layer in self.layers.iter() {
            inputs = layer.forward(&inputs);
        }
        inputs
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }
}
