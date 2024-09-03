use std::cell::RefCell;

use rand::{distributions::uniform::SampleUniform, thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

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
