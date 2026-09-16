fn main() {
    // An extension module leaves the Python symbols to be resolved when the
    // interpreter loads it; macOS needs `-undefined dynamic_lookup` for that
    // (maturin adds it, a plain `cargo build --workspace` does not).
    pyo3_build_config::add_extension_module_link_args();
}
