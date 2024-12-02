use crate::{gen_range, Layer, Value, Value1d};

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

    pub fn forward(&self, input: &Value1d) -> Value1d {
        if !self.is_training {
            return input.clone();
        }
        let mut output = Value1d::zeros(input.len());
        for i in 0..input.len() {
            if gen_range(0.0, 1.0) >= self.rate {
                output[i] = input[i].clone();
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

pub struct RecurrentLayer {
    input_layer: Layer,
    hidden_layer: Layer,
    hidden_size: usize,
    dropout_layer: Option<DropoutLayer>,
}

impl RecurrentLayer {
    pub fn new(input_size: usize, hidden_size: usize, dropout_rate: f64) -> Self {
        let input_layer = Layer::new(input_size, hidden_size, false);
        let hidden_layer = Layer::new(hidden_size, hidden_size, false);
        let dropout_layer = if dropout_rate > 0.0 {
            Some(DropoutLayer::new(dropout_rate))
        } else {
            None
        };
        RecurrentLayer {
            input_layer,
            hidden_layer,
            hidden_size,
            dropout_layer,
        }
    }

    pub fn parameters(&self) -> Value1d {
        self.input_layer
            .parameters()
            .into_iter()
            .chain(self.hidden_layer.parameters())
            .collect()
    }

    pub fn set_weights(&mut self, weight: Value) {
        self.input_layer.set_weights(weight.clone());
        self.hidden_layer.set_weights(weight);
    }

    pub fn set_bias(&mut self, bias: Value) {
        self.input_layer.set_bias(bias.clone());
        self.hidden_layer.set_bias(bias);
    }

    pub fn train(&mut self) {
        if let Some(dropout_layer) = &mut self.dropout_layer {
            dropout_layer.train();
        }
    }

    pub fn eval(&mut self) {
        if let Some(dropout_layer) = &mut self.dropout_layer {
            dropout_layer.eval();
        }
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    pub fn forward(&self, input: &Value1d, state: Option<Value1d>) -> (Value1d, Option<Value1d>) {
        let hidden_state = match state {
            Some(state) => state.clone(),
            None => Value1d::zeros(self.hidden_size),
        };
        let input_state = self.input_layer.forward(input);
        let hidden_state = self.hidden_layer.forward(&hidden_state);
        let hidden_state = (hidden_state + input_state).tanh();
        let mut output = hidden_state.clone();
        if let Some(dropout_layer) = &self.dropout_layer {
            output = dropout_layer.forward(&output);
        }
        (output, Some(hidden_state))
    }
}

pub struct RNN {
    rlayers: Vec<RecurrentLayer>,
    output_layer: Layer,
}

impl RNN {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        num_layers: usize,
        dropout_rate: f64,
    ) -> Self {
        assert!(num_layers > 0);
        let mut rlayers = Vec::new();
        for i in 0..num_layers {
            let isize = if i == 0 { input_size } else { hidden_size };
            let drate = if i == num_layers - 1 {
                0.0
            } else {
                dropout_rate
            };
            rlayers.push(RecurrentLayer::new(isize, hidden_size, drate));
        }
        let output_layer = Layer::new(hidden_size, output_size, false);
        RNN {
            rlayers,
            output_layer,
        }
    }

    pub fn train(&mut self) {
        for layer in self.rlayers.iter_mut() {
            layer.train();
        }
    }

    pub fn eval(&mut self) {
        for layer in self.rlayers.iter_mut() {
            layer.eval();
        }
    }

    pub fn set_weights(&mut self, weight: Value) {
        for layer in self.rlayers.iter_mut() {
            layer.set_weights(weight.clone());
        }
        self.output_layer.set_weights(weight);
    }

    pub fn set_bias(&mut self, bias: Value) {
        for layer in self.rlayers.iter_mut() {
            layer.set_bias(bias.clone());
        }
        self.output_layer.set_bias(bias);
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }

    pub fn parameters(&self) -> Value1d {
        self.rlayers
            .iter()
            .map(|layer| layer.parameters())
            .chain(std::iter::once(self.output_layer.parameters()))
            .flatten()
            .collect()
    }

    pub fn forward(
        &self,
        input: &Vec<Value1d>,        // shape: (seq_len, features)
        state: Option<Vec<Value1d>>, // shape: (num_layers, hidden_size)
    ) -> (Vec<Value1d>, Option<Vec<Value1d>>) {
        let mut states = match state {
            None => vec![None; self.rlayers.len()],
            Some(state) => state.into_iter().map(|s| Some(s)).collect(),
        };
        assert_eq!(states.len(), self.rlayers.len());
        let mut outputs = Vec::new();
        for i in input {
            let mut new_states = Vec::new();
            let mut output = i.clone();
            for (layer, lstate) in self.rlayers.iter().zip(states.into_iter()) {
                let (new_output, new_state) = layer.forward(&output, lstate);
                output = new_output;
                new_states.push(new_state);
            }
            states = new_states;
            output = self.output_layer.forward(&output);
            outputs.push(output);
        }
        (
            outputs,
            Some(states.iter().map(|x| x.clone().unwrap()).collect()),
        )
    }
}

pub struct LstmLayer {
    inlayers: Vec<Layer>,
    hlayers: Vec<Layer>,
    hidden_size: usize,
    dropout_layer: Option<DropoutLayer>,
}

impl LstmLayer {
    pub fn new(input_size: usize, hidden_size: usize, dropout_rate: f64) -> Self {
        let mut inlayers = Vec::new();
        for _ in 0..4 {
            inlayers.push(Layer::new(input_size, hidden_size, false));
        }
        let mut hlayers = Vec::new();
        for _ in 0..4 {
            hlayers.push(Layer::new(hidden_size, hidden_size, false));
        }
        let dropout_layer = if dropout_rate > 0.0 {
            Some(DropoutLayer::new(dropout_rate))
        } else {
            None
        };
        let mut l = LstmLayer {
            inlayers,
            hlayers,
            hidden_size,
            dropout_layer,
        };
        l.init_parameters();
        l
    }

    pub fn init_parameters(&mut self) {
        let r = (1.0 / self.hidden_size as f64).sqrt();
        for layer in self.inlayers.iter_mut() {
            layer.set_weights(Value::from(gen_range(-r, r)));
            layer.set_bias(Value::from(gen_range(-r, r)));
        }
        for layer in self.hlayers.iter_mut() {
            layer.set_weights(Value::from(gen_range(-r, r)));
            layer.set_bias(Value::from(gen_range(-r, r)));
        }
    }

    pub fn set_weights(&mut self, weight: Value) {
        for layer in self.inlayers.iter_mut() {
            layer.set_weights(weight.clone());
        }
        for layer in self.hlayers.iter_mut() {
            layer.set_weights(weight.clone());
        }
    }

    pub fn set_bias(&mut self, bias: Value) {
        for layer in self.inlayers.iter_mut() {
            layer.set_bias(bias.clone());
        }
        for layer in self.hlayers.iter_mut() {
            layer.set_bias(bias.clone());
        }
    }

    pub fn parameters(&self) -> Value1d {
        self.inlayers
            .iter()
            .map(|layer| layer.parameters())
            .chain(self.hlayers.iter().map(|layer| layer.parameters()))
            .flatten()
            .collect()
    }

    pub fn train(&mut self) {
        if let Some(dropout_layer) = &mut self.dropout_layer {
            dropout_layer.train();
        }
    }

    pub fn eval(&mut self) {
        if let Some(dropout_layer) = &mut self.dropout_layer {
            dropout_layer.eval();
        }
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    pub fn forward(
        &self,
        input: &Value1d,
        state: Option<(Value1d, Value1d)>, // (hidden_state, cell_state)
    ) -> (Value1d, Option<(Value1d, Value1d)>) {
        let (hidden_state, cell_state) = match state {
            Some((hidden_state, cell_state)) => (hidden_state.clone(), cell_state.clone()),
            None => (
                Value1d::zeros(self.hidden_size),
                Value1d::zeros(self.hidden_size),
            ),
        };
        let forget_gate =
            (self.inlayers[0].forward(input) + self.hlayers[0].forward(&hidden_state)).sigmoid();
        let input_gate =
            (self.inlayers[1].forward(input) + self.hlayers[1].forward(&hidden_state)).sigmoid();
        let output_gate =
            (self.inlayers[2].forward(input) + self.hlayers[2].forward(&hidden_state)).sigmoid();
        let cell_gate =
            (self.inlayers[3].forward(input) + self.hlayers[3].forward(&hidden_state)).tanh();

        let cell_state = forget_gate * cell_state + input_gate * cell_gate;
        let hidden_state = output_gate * cell_state.tanh();
        let mut output = hidden_state.clone();
        if let Some(dropout_layer) = &self.dropout_layer {
            output = dropout_layer.forward(&output);
        }
        (output, Some((hidden_state, cell_state)))
    }
}

pub struct LSTM {
    rlayers: Vec<LstmLayer>,
    output_layer: Layer,
}

impl LSTM {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        num_layers: usize,
        dropout_rate: f64,
    ) -> Self {
        assert!(num_layers > 0);
        let mut rlayers = Vec::new();
        for i in 0..num_layers {
            let isize = if i == 0 { input_size } else { hidden_size };
            let drate = if i == num_layers - 1 {
                0.0
            } else {
                dropout_rate
            };
            rlayers.push(LstmLayer::new(isize, hidden_size, drate));
        }
        let output_layer = Layer::new(hidden_size, output_size, false);
        LSTM {
            rlayers,
            output_layer,
        }
    }

    pub fn train(&mut self) {
        for layer in self.rlayers.iter_mut() {
            layer.train();
        }
    }

    pub fn eval(&mut self) {
        for layer in self.rlayers.iter_mut() {
            layer.eval();
        }
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }

    pub fn set_weights(&mut self, weight: Value) {
        for layer in self.rlayers.iter_mut() {
            layer.set_weights(weight.clone());
        }
        self.output_layer.set_weights(weight);
    }

    pub fn set_bias(&mut self, bias: Value) {
        for layer in self.rlayers.iter_mut() {
            layer.set_bias(bias.clone());
        }
        self.output_layer.set_bias(bias);
    }

    pub fn parameters(&self) -> Value1d {
        self.rlayers
            .iter()
            .map(|layer| layer.parameters())
            .chain(std::iter::once(self.output_layer.parameters()))
            .flatten()
            .collect()
    }

    pub fn forward(
        &self,
        input: &Vec<Value1d>,                   // shape: (seq_len, features)
        state: Option<Vec<(Value1d, Value1d)>>, // shape: (num_layers, hidden_size)
    ) -> (Vec<Value1d>, Option<Vec<(Value1d, Value1d)>>) {
        let mut states = match state {
            None => vec![None; self.rlayers.len()],
            Some(state) => state.into_iter().map(|s| Some(s)).collect(),
        };
        assert_eq!(states.len(), self.rlayers.len());
        let mut outputs = Vec::new();
        for i in input {
            let mut new_states = Vec::new();
            let mut output = i.clone();
            for (layer, lstate) in self.rlayers.iter().zip(states.into_iter()) {
                let (new_output, new_state) = layer.forward(&output, lstate);
                output = new_output;
                new_states.push(new_state);
            }
            states = new_states;
            output = self.output_layer.forward(&output);
            outputs.push(output);
        }
        (
            outputs,
            Some(states.iter().map(|x| x.clone().unwrap()).collect()),
        )
    }
}
