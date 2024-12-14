use crate::{DropoutLayer, Layer, Value, Value1d, Value2d};

#[derive(Clone, Debug)]
pub struct Embedding {
    embedding: Layer,
    vocab_size: usize,
    scale: Value,
}

impl Embedding {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Self {
            embedding: Layer::new(vocab_size, d_model, false),
            scale: Value::from((d_model as f64).sqrt()),
            vocab_size: vocab_size,
        }
    }

    pub fn set_weights(&mut self, weights: Value) {
        self.embedding.set_weights(weights);
    }

    pub fn set_bias(&mut self, bias: Value) {
        self.embedding.set_bias(bias);
    }

    fn one_hot_encode(&self, indices: &Vec<usize>) -> Vec<Value1d> {
        assert!(indices.iter().all(|i| *i < self.vocab_size));
        indices
            .iter()
            .map(|i| {
                let mut one_hot = Value1d::zeros(self.vocab_size);
                one_hot[*i] = Value::from(1.0);
                one_hot
            })
            .collect()
    }

    pub fn forward(&self, indices: &Vec<usize>) -> Vec<Value1d> {
        self.one_hot_encode(indices)
            .iter()
            .map(|i| self.embedding.forward(i) * &self.scale)
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct LayerNorm {
    scale: Value1d,
    bias: Value1d,
    eps: Value,
}

impl LayerNorm {
    pub fn new(d_model: usize, eps: f64) -> Self {
        Self {
            scale: Value1d::ones(d_model),
            bias: Value1d::zeros(d_model),
            eps: Value::from(eps),
        }
    }

    pub fn forward(
        &self,
        inputs: &Vec<Value1d>, // shape: (seq_len, d_model)
    ) -> Vec<Value1d> {
        inputs
            .iter()
            .map(|i| {
                let mean = i.mean();
                let variance = i.variance();
                let norm = (i - mean) / (variance + &self.eps).sqrt();
                norm * &self.scale + &self.bias
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct PositionalEncoding {
    encoding: Vec<Value1d>,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let n = 10000.0_f64;
        let d_model_f = d_model as f64;
        let mut encoding = Vec::new();
        for pos in 0..max_len {
            let pos_f = pos as f64;
            let mut encoding_pos = Value1d::new();
            for i in 0..d_model {
                let i_f = i as f64;
                let angle = pos_f / n.powf((2.0 * (i_f / 2.0)) / d_model_f);
                if i % 2 == 0 {
                    encoding_pos.push(Value::from(angle.sin()));
                } else {
                    encoding_pos.push(Value::from(angle.cos()));
                }
            }
            encoding.push(encoding_pos);
        }
        Self { encoding }
    }

    pub fn forward(
        &self,
        inputs: &Vec<Value1d>, // shape: (seq_len, d_model)
    ) -> Vec<Value1d> {
        inputs
            .iter()
            .enumerate()
            .map(|(idx, i)| i + &self.encoding[idx])
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct FeedForward {
    linear1: Layer,
    linear2: Layer,
}

impl FeedForward {
    pub fn new(d_model: usize, dim_feedforward: usize) -> Self {
        Self {
            linear1: Layer::new(d_model, dim_feedforward, true),
            linear2: Layer::new(dim_feedforward, d_model, false),
        }
    }

    pub fn parameters(&self) -> Value1d {
        self.linear1
            .parameters()
            .iter()
            .chain(self.linear2.parameters().iter())
            .cloned()
            .collect()
    }

    pub fn set_weights(&mut self, weights: Value) {
        self.linear1.set_weights(weights.clone());
        self.linear2.set_weights(weights);
    }

    pub fn set_bias(&mut self, bias: Value) {
        self.linear1.set_bias(bias.clone());
        self.linear2.set_bias(bias);
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }

    pub fn forward(&self, inputs: &Vec<Value1d>) -> Vec<Value1d> {
        let inputs = inputs
            .iter()
            .map(|i| self.linear1.forward(i))
            .collect::<Vec<Value1d>>();
        inputs
            .iter()
            .map(|i| self.linear2.forward(i))
            .collect::<Vec<Value1d>>()
    }
}

#[derive(Clone, Debug)]
pub struct Attention {
    linear_key: Layer,
    linear_value: Layer,
    linear_query: Layer,
    scale: Value,
    dropout_layer: Option<DropoutLayer>,
}

impl Attention {
    pub fn new(d_model: usize, d_head: usize, dropout: f64) -> Self {
        let mut dropout_layer = None;
        if dropout > 0.0 {
            dropout_layer = Some(DropoutLayer::new(dropout));
        }
        Self {
            dropout_layer: dropout_layer,
            linear_key: Layer::new(d_model, d_head, false),
            linear_value: Layer::new(d_model, d_head, false),
            linear_query: Layer::new(d_model, d_head, false),
            scale: Value::from((d_head as f64).sqrt()),
        }
    }

    pub fn parameters(&self) -> Value1d {
        self.linear_key
            .parameters()
            .iter()
            .chain(self.linear_value.parameters().iter())
            .chain(self.linear_query.parameters().iter())
            .cloned()
            .collect()
    }

    pub fn set_weights(&mut self, weights: Value) {
        self.linear_key.set_weights(weights.clone());
        self.linear_value.set_weights(weights.clone());
        self.linear_query.set_weights(weights);
    }

    pub fn set_bias(&mut self, bias: Value) {
        self.linear_key.set_bias(bias.clone());
        self.linear_value.set_bias(bias.clone());
        self.linear_query.set_bias(bias);
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }

    pub fn forward(
        &self,
        query: &Vec<Value1d>,          // shape: (tgt_seq_len, d_model)
        key: &Vec<Value1d>,            // shape: (src_seq_len, d_model)
        value: &Vec<Value1d>,          // shape: (src_seq_len, d_model)
        mask: Option<&Vec<Vec<bool>>>, // shape: (tgt_seq_len, src_seq_len)
    ) -> (
        Vec<Value1d>, // shape: (tgt_seq_len, d_head)
        Vec<Value1d>, // shape: (tgt_seq_len, src_seq_len)
    ) {
        let projected_q = query
            .iter()
            .map(|i| self.linear_query.forward(i))
            .collect::<Vec<Value1d>>();
        let projected_k = key
            .iter()
            .map(|i| self.linear_key.forward(i))
            .collect::<Vec<Value1d>>();
        let projected_v = value
            .iter()
            .map(|i: &Value1d| self.linear_value.forward(i))
            .collect::<Vec<Value1d>>();
        self.scaled_dot_product_attention(&projected_q, &projected_k, &projected_v, mask)
    }

    fn scaled_dot_product_attention(
        &self,
        query: &Vec<Value1d>,          // shape: (tgt_seq_len, d_head)
        key: &Vec<Value1d>,            // shape: (src_seq_len, d_head)
        value: &Vec<Value1d>,          // shape: (src_seq_len, d_head)
        mask: Option<&Vec<Vec<bool>>>, // shape: (tgt_seq_len, src_seq_len)
    ) -> (
        Vec<Value1d>, // shape: (tgt_seq_len, d_head)
        Vec<Value1d>, // shape: (tgt_seq_len, src_seq_len)
    ) {
        let mut scores = Value2d::from(query).matmul(&Value2d::from(key).transpose()) / &self.scale;
        assert!(scores.shape() == (query.len(), key.len()));
        if let Some(m) = mask {
            // Replace masked scores with small values
            for i in 0..m.len() {
                for j in 0..m[i].len() {
                    if m[i][j] {
                        scores[(i, j)] = Value::from(-1e8);
                    }
                }
            }
        }
        let mut attention_weights = scores.softmax_axis_1();
        if let Some(d) = &self.dropout_layer {
            attention_weights = attention_weights
                .to_value1d()
                .iter()
                .map(|i| d.forward(i))
                .collect();
        }
        (
            attention_weights.matmul(&Value2d::from(value)).to_value1d(),
            attention_weights.to_value1d(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct MultiheadAttention {
    heads: Vec<Attention>,
    output_layer: Layer,
}

impl MultiheadAttention {
    pub fn new(d_model: usize, num_heads: usize, dropout: f64) -> Self {
        assert!(d_model % num_heads == 0);
        let d_head = d_model / num_heads;
        let mut heads = Vec::new();
        for _ in 0..num_heads {
            heads.push(Attention::new(d_model, d_head, dropout));
        }
        Self {
            heads: heads,
            output_layer: Layer::new(d_model, d_model, false),
        }
    }

    pub fn parameters(&self) -> Value1d {
        self.heads
            .iter()
            .map(|i| i.parameters())
            .chain(std::iter::once(self.output_layer.parameters()))
            .flatten()
            .collect()
    }

    pub fn set_weights(&mut self, weights: Value) {
        for h in &mut self.heads {
            h.set_weights(weights.clone());
        }
        self.output_layer.set_weights(weights);
    }

    pub fn set_bias(&mut self, bias: Value) {
        for h in &mut self.heads {
            h.set_bias(bias.clone());
        }
        self.output_layer.set_bias(bias);
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }

    pub fn forward(
        &self,
        query: &Vec<Value1d>,          // shape: (tgt_seq_len, d_model)
        key: &Vec<Value1d>,            // shape: (src_seq_len, d_model)
        value: &Vec<Value1d>,          // shape: (src_seq_len, d_model)
        mask: Option<&Vec<Vec<bool>>>, // shape: (tgt_seq_len, src_seq_len)
    ) -> (
        Vec<Value1d>, // shape: (tgt_seq_len, d_model)
        Vec<Value1d>, // shape: (tgt_seq_len, src_seq_len)
    ) {
        let mut head_attentions = (0..query.len())
            .map(|_| Value1d::new())
            .collect::<Vec<Value1d>>();
        let mut head_weights = Value2d::zeros((query.len(), key.len()));
        for h in &self.heads {
            let (attention, weights) = h.forward(query, key, value, mask);
            head_weights += Value2d::from(weights);
            for (seq_idx, a) in attention.into_iter().enumerate() {
                head_attentions[seq_idx].extend(a);
            }
        }
        let output = head_attentions
            .iter()
            .map(|i| self.output_layer.forward(i))
            .collect();
        let avg_weights = head_weights / Value::from(self.heads.len() as f64);
        (output, avg_weights.to_value1d())
    }
}

#[derive(Clone, Debug)]
pub struct DecoderLayer {
    multihead_attention1: MultiheadAttention,
    multihead_attention2: MultiheadAttention,
    feedforward: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    layer_norm3: LayerNorm,
    dropout1: DropoutLayer,
    dropout2: DropoutLayer,
    dropout3: DropoutLayer,
}

impl DecoderLayer {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        dim_feedforward: usize,
        dropout: f64,
        layer_norm_eps: f64,
    ) -> Self {
        Self {
            multihead_attention1: MultiheadAttention::new(d_model, num_heads, dropout),
            multihead_attention2: MultiheadAttention::new(d_model, num_heads, dropout),
            feedforward: FeedForward::new(d_model, dim_feedforward),
            layer_norm1: LayerNorm::new(d_model, layer_norm_eps),
            layer_norm2: LayerNorm::new(d_model, layer_norm_eps),
            layer_norm3: LayerNorm::new(d_model, layer_norm_eps),
            dropout1: DropoutLayer::new(dropout),
            dropout2: DropoutLayer::new(dropout),
            dropout3: DropoutLayer::new(dropout),
        }
    }

    pub fn parameters(&self) -> Value1d {
        self.multihead_attention1
            .parameters()
            .iter()
            .chain(self.multihead_attention2.parameters().iter())
            .chain(self.feedforward.parameters().iter())
            .cloned()
            .collect()
    }

    pub fn set_weights(&mut self, weights: Value) {
        self.multihead_attention1.set_weights(weights.clone());
        self.multihead_attention2.set_weights(weights.clone());
        self.feedforward.set_weights(weights);
    }

    pub fn set_bias(&mut self, bias: Value) {
        self.multihead_attention1.set_bias(bias.clone());
        self.multihead_attention2.set_bias(bias.clone());
        self.feedforward.set_bias(bias);
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }

    pub fn forward(
        &self,
        tgt: &Vec<Value1d>,                  // shape: (tgt_seq_len, d_model)
        memory: &Vec<Value1d>,               // shape: (seq_len, d_model)
        tgt_mask: Option<Vec<Vec<bool>>>,    // shape: (tgt_seq_len, tgt_seq_len)
        memory_mask: Option<Vec<Vec<bool>>>, // shape: (tgt_seq_len, src_seq_len)
    ) -> Vec<Value1d> // shape: (tgt_seq_len, d_model)
    {
        let (attention1, _) = self
            .multihead_attention1
            .forward(tgt, tgt, tgt, tgt_mask.as_ref());
        let attention1 = attention1
            .iter()
            .zip(tgt.iter())
            .map(|(a, s)| (self.dropout1.forward(a) + s))
            .collect::<Vec<Value1d>>();
        let attention1 = self.layer_norm1.forward(&attention1);
        let (attention2, _) =
            self.multihead_attention2
                .forward(&attention1, memory, memory, memory_mask.as_ref());
        let attention2 = attention2
            .iter()
            .zip(attention1.iter())
            .map(|(a, s)| (self.dropout2.forward(a) + s))
            .collect::<Vec<Value1d>>();
        let attention2 = self.layer_norm2.forward(&attention2);
        let attention2 = self
            .feedforward
            .forward(&attention2)
            .iter()
            .zip(attention2.iter())
            .map(|(f, a)| (self.dropout3.forward(f) + a))
            .collect::<Vec<Value1d>>();
        self.layer_norm3.forward(&attention2)
    }
}

#[derive(Clone, Debug)]
pub struct Decoder {
    layers: Vec<DecoderLayer>,
}

impl Decoder {
    pub fn new(layer: DecoderLayer, num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| layer.clone()).collect(),
        }
    }

    pub fn parameters(&self) -> Value1d {
        self.layers
            .iter()
            .map(|i| i.parameters())
            .flatten()
            .collect()
    }

    pub fn set_weights(&mut self, weights: Value) {
        for l in &mut self.layers {
            l.set_weights(weights.clone());
        }
    }

    pub fn set_bias(&mut self, bias: Value) {
        for l in &mut self.layers {
            l.set_bias(bias.clone());
        }
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }

    pub fn forward(
        &self,
        tgt: &Vec<Value1d>,                  // shape: (tgt_seq_len, d_model)
        memory: &Vec<Value1d>,               // shape: (src_seq_len, d_model)
        tgt_mask: Option<Vec<Vec<bool>>>,    // shape: (tgt_seq_len, tgt_seq_len)
        memory_mask: Option<Vec<Vec<bool>>>, // shape: (tgt_seq_len, src_seq_len)
    ) -> Vec<Value1d> // shape: (tgt_seq_len, d_model)
    {
        let mut output = tgt.clone();
        for layer in &self.layers {
            output = layer.forward(&output, memory, tgt_mask.clone(), memory_mask.clone());
        }
        output
    }
}

#[derive(Clone, Debug)]
pub struct EncoderLayer {
    multihead_attention: MultiheadAttention,
    feedforward: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    dropout1: DropoutLayer,
    dropout2: DropoutLayer,
}

impl EncoderLayer {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        dim_feedforward: usize,
        dropout: f64,
        layer_norm_eps: f64,
    ) -> Self {
        Self {
            multihead_attention: MultiheadAttention::new(d_model, num_heads, dropout),
            feedforward: FeedForward::new(d_model, dim_feedforward),
            layer_norm1: LayerNorm::new(d_model, layer_norm_eps),
            layer_norm2: LayerNorm::new(d_model, layer_norm_eps),
            dropout1: DropoutLayer::new(dropout),
            dropout2: DropoutLayer::new(dropout),
        }
    }

    pub fn parameters(&self) -> Value1d {
        self.multihead_attention
            .parameters()
            .iter()
            .chain(self.feedforward.parameters().iter())
            .cloned()
            .collect()
    }

    pub fn set_weights(&mut self, weights: Value) {
        self.multihead_attention.set_weights(weights.clone());
        self.feedforward.set_weights(weights);
    }

    pub fn set_bias(&mut self, bias: Value) {
        self.multihead_attention.set_bias(bias.clone());
        self.feedforward.set_bias(bias);
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }

    pub fn forward(
        &self,
        src: &Vec<Value1d>,               // shape: (src_seq_len, d_model)
        src_mask: Option<Vec<Vec<bool>>>, // shape: (src_seq_len, src_seq_len)
    ) -> Vec<Value1d> // shape: (src_seq_len, d_model)
    {
        let (attention, _) = self
            .multihead_attention
            .forward(src, src, src, src_mask.as_ref());
        let attention = attention
            .iter()
            .zip(src.iter())
            .map(|(a, s)| (self.dropout1.forward(a) + s))
            .collect::<Vec<Value1d>>();
        let attention = self.layer_norm1.forward(&attention);
        let attention = self
            .feedforward
            .forward(&attention)
            .iter()
            .zip(attention.iter())
            .map(|(f, a)| (self.dropout2.forward(f) + a))
            .collect::<Vec<Value1d>>();
        self.layer_norm2.forward(&attention)
    }
}

#[derive(Clone, Debug)]
pub struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    pub fn new(layer: EncoderLayer, num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| layer.clone()).collect(),
        }
    }

    pub fn parameters(&self) -> Value1d {
        self.layers
            .iter()
            .map(|i| i.parameters())
            .flatten()
            .collect()
    }

    pub fn set_weights(&mut self, weights: Value) {
        for l in &mut self.layers {
            l.set_weights(weights.clone());
        }
    }

    pub fn set_bias(&mut self, bias: Value) {
        for l in &mut self.layers {
            l.set_bias(bias.clone());
        }
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }

    pub fn forward(
        &self,
        src: &Vec<Value1d>,               // shape: (src_seq_len, d_model)
        src_mask: Option<Vec<Vec<bool>>>, // shape: (src_seq_len, src_seq_len)
    ) -> Vec<Value1d> // shape: (src_seq_len, d_model)
    {
        let mut output = src.clone();
        for layer in &self.layers {
            output = layer.forward(src, src_mask.clone());
        }
        output
    }
}

#[derive(Clone, Debug)]
pub struct Transformer {
    encoder: Encoder,
    decoder: Decoder,
}

impl Transformer {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        dim_feedforward: usize,
        dropout: f64,
        layer_norm_eps: f64,
    ) -> Self {
        let encoder_layer =
            EncoderLayer::new(d_model, num_heads, dim_feedforward, dropout, layer_norm_eps);
        let decoder_layer =
            DecoderLayer::new(d_model, num_heads, dim_feedforward, dropout, layer_norm_eps);
        Self {
            encoder: Encoder::new(encoder_layer, num_encoder_layers),
            decoder: Decoder::new(decoder_layer, num_decoder_layers),
        }
    }

    pub fn parameters(&self) -> Value1d {
        self.encoder
            .parameters()
            .iter()
            .chain(self.decoder.parameters().iter())
            .cloned()
            .collect()
    }

    pub fn set_weights(&mut self, weights: Value) {
        self.encoder.set_weights(weights.clone());
        self.decoder.set_weights(weights);
    }

    pub fn set_bias(&mut self, bias: Value) {
        self.encoder.set_bias(bias.clone());
        self.decoder.set_bias(bias);
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }

    pub fn forward(
        &self,
        src: &Vec<Value1d>,                  // shape: (src_seq_len, d_model)
        tgt: &Vec<Value1d>,                  // shape: (tgt_seq_len, d_model)
        src_mask: Option<Vec<Vec<bool>>>,    // shape: (src_seq_len, src_seq_len)
        tgt_mask: Option<Vec<Vec<bool>>>,    // shape: (tgt_seq_len, tgt_seq_len)
        memory_mask: Option<Vec<Vec<bool>>>, // shape: (tgt_seq_len, src_seq_len)
    ) -> Vec<Value1d> // shape: (tgt_seq_len, d_model)
    {
        let memory = self.encoder.forward(src, src_mask);
        self.decoder.forward(tgt, &memory, tgt_mask, memory_mask)
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    transformer: Transformer,
    src_embed: Embedding,
    tgt_embed: Embedding,
    position_encoding: PositionalEncoding,
    linear: Layer,
}

impl Model {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_heads: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        dim_feedforward: usize,
        dropout: f64,
        layer_norm_eps: f64,
        max_len: usize,
    ) -> Self {
        let transformer = Transformer::new(
            d_model,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            layer_norm_eps,
        );
        let src_embed = Embedding::new(vocab_size, d_model);
        let tgt_embed = Embedding::new(vocab_size, d_model);
        let position_encoding = PositionalEncoding::new(d_model, max_len);
        let linear = Layer::new(d_model, vocab_size, false);
        Self {
            transformer,
            src_embed,
            tgt_embed,
            position_encoding,
            linear: linear,
        }
    }

    pub fn parameters(&self) -> Value1d {
        self.transformer
            .parameters()
            .iter()
            .chain(self.linear.parameters().iter())
            .cloned()
            .collect()
    }

    pub fn set_weights(&mut self, weights: Value) {
        self.transformer.set_weights(weights.clone());
        self.linear.set_weights(weights);
    }

    pub fn set_bias(&mut self, bias: Value) {
        self.transformer.set_bias(bias.clone());
        self.linear.set_bias(bias);
    }

    pub fn zero_grad(&self) {
        self.parameters().zero_grad();
    }

    pub fn update(&self, rate: f64) {
        self.parameters().update(rate);
    }

    pub fn forward(
        &self,
        src: &Vec<usize>, // shape: (src_seq_len)
        tgt: &Vec<usize>, // shape: (tgt_seq_len)
    ) -> Vec<Value1d> // shape: (tgt_seq_len)
    {
        let src_embed = self.position_encoding.forward(&self.src_embed.forward(src));
        let tgt_embed = self.position_encoding.forward(&self.tgt_embed.forward(tgt));
        let output = self
            .transformer
            .forward(&src_embed, &tgt_embed, None, None, None);
        output
            .iter()
            .map(|i| self.linear.forward(i).softmax())
            .collect()
    }
}
