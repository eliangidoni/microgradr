use crate::{Layer, Value, Value1d, Value2d};

pub struct Dropout2dLayer {
    rate: f64,
    is_training: bool,
}

impl Dropout2dLayer {
    pub fn new(rate: f64) -> Self {
        assert!(rate >= 0.0 && rate <= 1.0);
        Self {
            rate,
            is_training: true,
        }
    }

    // inputs shape: (channels, height, width)
    pub fn forward(&self, inputs: &Vec<Value2d>) -> Vec<Value2d> {
        if !self.is_training {
            return inputs.clone();
        }
        let mut outputs = (0..inputs.len())
            .map(|i| Value2d::zeros(inputs[i].shape()))
            .collect::<Vec<Value2d>>();
        for c in 0..inputs.len() {
            for i in 0..inputs[c].shape().0 {
                for j in 0..inputs[c].shape().1 {
                    if crate::gen_range(0.0, 1.0) >= self.rate {
                        outputs[c][(i, j)] = inputs[c][(i, j)].clone();
                    }
                }
            }
        }
        outputs
    }

    pub fn train(&mut self) {
        self.is_training = true;
    }

    pub fn eval(&mut self) {
        self.is_training = false;
    }
}

pub struct MaxPooling2dLayer {
    kernel_size: usize,
    stride: usize,
}

impl MaxPooling2dLayer {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
        }
    }

    // inputs shape: (channels, height, width)
    pub fn forward(&self, inputs: &Vec<Value2d>) -> Vec<Value2d> {
        let mut outputs = (0..inputs.len())
            .map(|i| {
                Value2d::zeros((
                    (inputs[i].shape().0 - self.kernel_size) / self.stride + 1,
                    (inputs[i].shape().1 - self.kernel_size) / self.stride + 1,
                ))
            })
            .collect::<Vec<Value2d>>();
        for c in 0..inputs.len() {
            for i in 0..outputs[c].shape().0 {
                for j in 0..outputs[c].shape().1 {
                    let mut max_value = Value::from(f64::NEG_INFINITY);
                    for k in 0..self.kernel_size {
                        for l in 0..self.kernel_size {
                            let value = &inputs[c][(i * self.stride + k, j * self.stride + l)];
                            if *value > max_value {
                                max_value = value.clone();
                            }
                        }
                    }
                    outputs[c][(i, j)] = max_value;
                }
            }
        }
        outputs
    }
}

pub struct Conv2dLayer {
    kernel_size: usize,
    in_channels: usize,
    out_channels: usize,
    stride: usize,
    filters: Vec<Vec<Value2d>>,
    biases: Value1d,
}

impl Conv2dLayer {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize) -> Self {
        let filters = (0..out_channels)
            .map(|_| {
                (0..in_channels)
                    .map(|_| Value2d::rand((kernel_size, kernel_size)))
                    .collect()
            })
            .collect();
        let biases = Value1d::rand(out_channels);
        Self {
            kernel_size,
            in_channels,
            out_channels,
            stride,
            filters,
            biases,
        }
    }

    pub fn parameters(&self) -> Value1d {
        self.filters
            .iter()
            .flat_map(|filter| filter.iter())
            .flat_map(|filter| filter.iter())
            .chain(self.biases.iter())
            .cloned()
            .collect()
    }

    pub fn set_weights(&mut self, weights: Value1d) {
        assert_eq!(weights.len(), self.parameters().len() - self.biases.len());
        let mut index = 0;
        for i in 0..self.out_channels {
            for j in 0..self.in_channels {
                for k in 0..self.kernel_size {
                    for l in 0..self.kernel_size {
                        self.filters[i][j][(k, l)] = weights[index].clone();
                        index += 1;
                    }
                }
            }
        }
    }

    pub fn set_biases(&mut self, biases: Value1d) {
        assert_eq!(biases.len(), self.biases.len());
        self.biases = biases.clone();
    }

    // inputs shape: (channels, height, width)
    pub fn forward(&self, inputs: &Vec<Value2d>) -> Vec<Value2d> {
        assert_eq!(inputs.len(), self.in_channels);
        let mut outputs = (0..self.out_channels)
            .map(|_| {
                Value2d::zeros((
                    (inputs[0].shape().0 - self.kernel_size) / self.stride + 1,
                    (inputs[0].shape().1 - self.kernel_size) / self.stride + 1,
                ))
            })
            .collect::<Vec<Value2d>>();
        for outc in 0..self.out_channels {
            for i in 0..outputs[outc].shape().0 {
                for j in 0..outputs[outc].shape().1 {
                    for inc in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            for l in 0..self.kernel_size {
                                outputs[outc][(i, j)] = &outputs[outc][(i, j)]
                                    + &inputs[inc][(i * self.stride + k, j * self.stride + l)]
                                        * &self.filters[outc][inc][(k, l)];
                            }
                        }
                    }
                    outputs[outc][(i, j)] = &outputs[outc][(i, j)] + &self.biases[outc];
                }
            }
        }
        outputs
    }
}

pub struct CNN {
    conv: Conv2dLayer,
    dropout: Dropout2dLayer,
    maxpooling: MaxPooling2dLayer,
    layer: Layer,
    out_channels: usize,
}

impl CNN {
    pub fn new(in_channels: usize, height: usize, width: usize, outputs: usize) -> Self {
        assert!(height >= 3 && width >= 3);
        let out_channels = 4;
        let conv = Conv2dLayer::new(in_channels, out_channels, 2, 1);
        let dropout = Dropout2dLayer::new(0.1);
        let maxpooling = MaxPooling2dLayer::new(2, 1);
        let layer = Layer::new(4 * (height - 2) * (width - 2), outputs, false);
        Self {
            conv,
            dropout,
            maxpooling,
            layer,
            out_channels,
        }
    }

    pub fn eval(&mut self) {
        self.dropout.eval();
    }

    pub fn train(&mut self) {
        self.dropout.train();
    }

    pub fn parameters(&self) -> Value1d {
        self.conv
            .parameters()
            .iter()
            .chain(self.layer.parameters().iter())
            .cloned()
            .collect()
    }

    pub fn set_bias(&mut self, bias: Value) {
        let mut biases = Value1d::zeros(self.out_channels);
        for i in 0..biases.len() {
            biases[i] = bias.clone();
        }
        self.conv.set_biases(biases);
        self.layer.set_bias(bias);
    }

    pub fn set_weights(&mut self, weight: Value) {
        let mut weights = Value1d::zeros(self.conv.parameters().len() - self.out_channels);
        for i in 0..weights.len() {
            weights[i] = weight.clone();
        }
        self.conv.set_weights(weights);
        self.layer.set_weights(weight);
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    // inputs shape: (channels, height, width)
    pub fn forward(&self, mut inputs: Vec<Value2d>) -> Value1d {
        inputs = self
            .conv
            .forward(&inputs)
            .iter()
            .map(|input| input.relu())
            .collect();
        inputs = self.dropout.forward(&inputs);
        inputs = self.maxpooling.forward(&inputs);
        let inputs_flattened = inputs.iter().flatten().cloned().collect::<Value1d>();
        self.layer.forward(&inputs_flattened)
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }
}
