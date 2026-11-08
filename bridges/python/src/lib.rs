//! A Python bridge for an external RL trainer over the `market-maker-rs` RL spaces.
//!
//! [`worker::Worker`] runs one environment on its own thread, which owns the dataset
//! and borrows the kernel inside its own function, so no lifetime crosses into Python.
//! Requests and replies go over `std` channels. The feature `python` adds the extension
//! module on top; the worker builds and tests without Python.

pub mod maintenance;
pub mod reactive;
pub mod worker;

#[cfg(feature = "python")]
mod python;
