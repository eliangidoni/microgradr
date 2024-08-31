pub mod engine;
pub use engine::{Value, ValueVec, ValueVecIter};
pub mod nn;
pub use nn::{set_seed, gen_range, Layer, Neuron, MLP};
pub mod sgd;
pub use sgd::mean_squared_error;

#[macro_export]
macro_rules! v {
    ($x:expr) => {
        &microgradr::Value::from($x)
    };
}

#[macro_export]
macro_rules! vs {
    ($( $x:expr ),*) => {
        {
            let mut tmp = microgradr::ValueVec::new();
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
