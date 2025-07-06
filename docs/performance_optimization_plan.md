# ⚡ Performance Optimization Plan for ClawFoxyVision (June 2025)

> This document captures **planned changes** to increase execution speed and overall efficiency on machines that provide an NVIDIA GPU with CUDA support.  It covers build-time tweaks, runtime device selection, data-pipeline optimisation, and profiling/benchmarking strategy.  Implementation PRs will reference the section numbers below.

---

## 1  Compile-Time Tweaks (Quick Wins)

1.1  **Release Flags** – Always build artefacts used in production or benchmarking with
```bash
cargo build --release
```
1.2  **Per-crate RUSTFLAGS** – add `.cargo/config.toml`:
```toml
[profile.release]
codegen-units = 1          # better optimisation
lto            = true       # link-time optimisation
opt-level      = "z"        # or "3" for max speed depending on binary size requirements
[env]
RUSTFLAGS="-C target-cpu=native"
```

## 2  Enable GPU Acceleration (No LibTorch)

2.1  **Switch to WGPU Backend**  
Burn 0.17 ships a cross-platform GPU backend powered by **wgpu** that does **not** require LibTorch or CUDA drivers.
To enable it:
```toml
# Cargo.toml
burn = { version = "0.17", features = [ "wgpu", "ndarray", "train" ] }
# Remove the explicit burn-tch dependency
```

2.2  **Runtime Device Helper** – update util to pick WGPU when present:
```rust
pub enum ComputeDevice {
    Cpu(NdArrayDevice),
    Gpu(WgpuDevice),   // from burn_wgpu
}

pub fn best_available() -> ComputeDevice {
    if let Some(dev) = WgpuDevice::default().ok() {  // returns Err if no compatible adapter
        ComputeDevice::Gpu(dev)
    } else {
        ComputeDevice::Cpu(NdArrayDevice::default())
    }
}
```

2.3  **GPU-ready Examples** – add new examples (`examples/wgpu_*`) showing model training/inference on the WGPU device.

2.4  **WebGPU-specific Tweaks**
   • Prefer WGSL `f16` where precision allows → halves memory and bandwidth (enable `shader-f16` on adapters that expose it).  
   • Re-use `GPURenderPipeline` objects – creating pipelines each frame is extremely expensive; cache by a deterministic key.  
   • Batch resource updates via `queue.write_buffer`/`write_texture` outside render pass.  
   • Keep bind-group count low (<4) and prefer dynamic offsets instead of thousands of distinct bind groups.

2.5  **Operation Fusion & Burn JIT**  
Burn 0.18 introduces optional kernel fusion on the WGPU backend.  Compile with:
```bash
cargo add burn --features "wgpu,fusion"
```
Set `BURN_WGPU_FUSION=1` at runtime to enable automatic op fusion for supported patterns (matmul-bias-activation, reduction chains, element-wise combos).

## 3  Data-Pipeline Optimisation (Polars)

3.1  **LazyFrames Everywhere** – read CSV/Parquet via `LazyFrame::scan_csv` / `scan_parquet`, apply transforms, `.collect()` once.

3.2  **SIMD & Performant Features** – extend `polars` dependency:
```toml
polars = { version = "0.47", features = [ "lazy", "rolling_window", "temporal", "performant", "simd" ] }
```

3.3  **Avoid Materialising Unused Columns** – select only required columns before `.collect()`.

3.4  **OHLCV-centric Optimisations**  
   • **Column Projection Early** – all loaders must immediately project to `["open","high","low","close","volume"]` (and any engineered columns) to minimise IO.  
   • **Parquet First** – nightly ETL converts every raw CSV in `examples/csv/` to column-pruned Parquet using `polars::frame_to_parquet` (≈ 5-10× faster scan).  
   • **Caching Windowed Sequences** – persist pre-windowed tensors (shape `[batch, seq_len, 5]`) per ticker/date-range to `${CACHE_DIR}/tensors/` to skip repeat slicing during hyper-parameter sweeps.

## 4  Parallelism & rayon

4.1  **Batch Generation** – replace manual `for` loops in `step_1_tensor_preparation` with `par_iter_mut` slices.

4.2  **Technical Indicators** – use rayon to parallelise per-column computations in `util::feature_engineering`.

4.3  **Technical-Indicator Kernels** – port SMA, EMA, RSI, MACD, Bollinger Bands to `burn::tensor` ops and expose optional WGPU kernels; parallelise remaining CPU path with Rayon `par_iter` across columns.

## 5  RNN Kernel Efficiency

5.1  **Remove Per-time-step Rust loops** – leverage Burn's built-in `LstmConfig` / `GruConfig` layers (which call fused kernels in LibTorch) instead of custom `for t in 0..seq_len` loops inside `*_lstm_cell.rs` and `*_gru_cell.rs`.

5.2  **Bidirectional Concatenation** – replace manual per-timestep `slice_assign` with `Tensor::cat` on complete sequences where possible.

## 6  Memory Optimisation

6.1  Use `Tensor::zeros_like` and in-place ops (`*_`) wherever Burn allows; avoid redundant clones.

6.2  Prefer `Arc<Tensor>` in data loaders to share immutable batches.

## 7  Profiling & Benchmarking

7.1  Baseline scripts in `scripts/profile_gpu.sh` using:
```bash
hyperfine --warmup 3 'cargo run --release -- AAPL lstm'
```
7.2  GPU kernels – capture with `nvprof` / `nsys` on Linux; CPU call-graphs via `cargo flamegraph`.

7.3  Set regression guard: CI fails if **training epoch-time** increases by >5 % wrt previous main.

## 8  Timeline & Ownership

| Milestone | Owner | PRs |
|-----------|-------|-----|
| CUDA build doc & `.cargo/config.toml` | `@devA` | #TBD |
| Device abstraction + refactor | `@devB` | #TBD |
| Data-pipeline lazy rewrite | `@data` | #TBD |
| RNN cell refactor to fused ops | `@ml` | #TBD |
| Benchmark harness & CI gate | `@ops` | #TBD |

## 9  Mixed-Precision & Quantization

9.1  **Automatic fp16 / bf16** – When using WGPU or CUDA backends, enable `mixed_precision` training/inference to reduce memory footprint and boost tensor-core throughput:
```rust
let config = TrainingConfig { mixed_precision: Some(MixedPrecision::Fp16), ..Default::default() };
```
9.2  **Post-training Dynamic Quantization** – For inference-only models ship 8-bit linear layers using `burn::nn::quant` (new in 0.18).

## 10  Advanced Profiling & Observability

10.1  **Flamegraphs** – add a `scripts/profile_cpu.sh` helper:
```bash
cargo flamegraph --example lstm_example
```
10.2  **GPU Timestamp Queries** – wrap training step in `PerformanceMonitor` util (see Burn blog Jul-2025). Use `BURN_WGPU_TIMESTAMPS=1` to gather per-kernel timings.

10.3  **Continuous Benchmarks** – integrate `criterion.rs` micro-benchmarks under `benches/` and run via CI.  Guard with `cargo critcmp` regression threshold (<5 %).

10.4  **Runtime Metrics** – expose Prometheus endpoint with model latency/histograms using `metrics` crate + `metrics-exporter-prometheus`.

---

**Next steps:** raise individual issues for each section; link them back to this plan. Once merged, update `docs/technical-reference.md` with new backend architecture and add GPU benchmark numbers to `docs/user-guide.md`. 

*This document is living; revisit quarterly as Burn & WebGPU evolve.* 