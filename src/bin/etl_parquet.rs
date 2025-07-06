use polars::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};

/// Columns to keep in Parquet for OHLCV workload
const BASE_COLS: &[&str] = &["open", "high", "low", "close", "volume"];

fn main() -> PolarsResult<()> {
    let csv_dir = PathBuf::from("examples/csv");
    let entries = fs::read_dir(&csv_dir).expect("Unable to read examples/csv directory");

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map(|e| e == "csv").unwrap_or(false) {
            println!("Processing {:?}", path.file_name().unwrap());
            convert_csv_to_parquet(&path)?;
        }
    }

    Ok(())
}

fn convert_csv_to_parquet(csv_path: &Path) -> PolarsResult<()> {
    let filename_stem = csv_path.file_stem().unwrap().to_string_lossy();
    let parquet_path = csv_path
        .parent()
        .unwrap()
        .join(format!("{}.parquet", filename_stem));

    // Lazy CSV scan with early projection
    let lf = LazyCsvReader::new(csv_path.to_string_lossy().as_ref())
        .has_header(true)
        .finish()?;

    // Ensure columns exist then project
    let lf = lf.select(
        BASE_COLS
            .iter()
            .filter_map(|&c| lf.schema().get_field(c).map(|_| col(c)))
            .collect::<Vec<_>>(),
    );

    let df = lf.collect()?;
    ParquetWriter::new(parquet_path).finish(&df)?;
    Ok(())
} 