use criterion::{criterion_group, criterion_main, Criterion};
use polars::prelude::*;
use clawfoxyvision::minute::lstm::step_1_tensor_preparation::dataframe_to_tensors; // adjust path

fn bench_seq_prep(c: &mut Criterion) {
    // load sample parquet
    let df = ParquetReader::new("examples/csv/AAPL_daily_ohlcv.parquet").finish().unwrap();
    type B = burn_ndarray::NdArray<f32>;
    let device = burn_ndarray::NdArrayDevice::default();

    c.bench_function("seq_prep_60", |b| {
        b.iter(|| {
            let _ = dataframe_to_tensors::<B>(&df, 60, 1, &device, false, Some(256)).unwrap();
        });
    });
}

criterion_group!(benches, bench_seq_prep);
criterion_main!(benches); 