pub mod engine;
pub use engine::{Value, Value1d, Value1dIter, Value2d};
pub mod cnn;
pub mod gnn;
pub mod nn;
pub mod rand;
pub mod rnn;
pub use cnn::{Conv2dLayer, Dropout2dLayer, MaxPooling2dLayer, CNN};
pub use nn::{Layer, Neuron, MLP};
pub use rand::{gen_range, set_seed};
pub use rnn::{DropoutLayer, LstmLayer, RecurrentLayer, LSTM, RNN};
pub mod sgd;
pub use sgd::mean_squared_error;
pub mod gbm;
pub mod transformer;
pub use transformer::{
    Attention, Decoder, DecoderLayer, Embedding, Encoder, EncoderLayer, FeedForward, LayerNorm,
    Model, MultiheadAttention, PositionalEncoding, Transformer,
};
pub mod tree;
pub use tree::Tree;

#[macro_export]
macro_rules! v {
    ($x:expr) => {
        &microgradr::Value::from($x)
    };
}

#[macro_export]
macro_rules! v1d {
    ($( $x:expr ),*) => {
        {
            let mut tmp = microgradr::Value1d::new();
            $(
                tmp.push(microgradr::Value::from($x));
            )*
            tmp
        }
    };
}

#[macro_export]
macro_rules! vvec {
    ($( $x:expr ),*) => {
        {
            let mut tmp = Vec::new();
            $(
                tmp.push(microgradr::Value::from($x));
            )*
            tmp
        }
    };
}
