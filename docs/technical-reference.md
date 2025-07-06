# ClawFoxyVision Technical Reference

This document provides detailed technical information about ClawFoxyVision's implementation, architecture, and algorithms.

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Input    │    │  Preprocessing  │    │   Model Layer   │
│                 │    │                 │    │                 │
│ • CSV/Parquet   │───▶│ • Normalization │───▶│ • LSTM/GRU/CNN  │
│ • OHLC Data     │    │ • Feature Eng.  │    │ • Training      │
│ • Technical     │    │ • Sequence Gen. │    │ • Prediction    │
│   Indicators    │    │ • Validation    │    │ • Serialization │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Output Layer  │
                       │                 │
                       │ • Predictions   │
                       │ • Metrics       │
                       │ • Model Files   │
                       └─────────────────┘
```

### Backend Architecture

ClawFoxyVision uses the **Burn** framework with **NdArray** backend:

```rust
type BurnBackend = Autodiff<NdArray<f32>>;
let device = NdArrayDevice::default();
```

**Key Components:**
- **Autodiff**: Automatic differentiation for gradient computation
- **NdArray**: CPU-based tensor operations
- **Device**: CPU device for computation

## 🧠 Model Architectures

### LSTM (Long Short-Term Memory)

#### Architecture Overview

```rust
pub struct LstmModel<B: Backend> {
    lstm: Lstm<B>,
    dropout: Dropout,
    linear: Linear<B>,
}
```

#### Cell Implementation

```rust
pub struct Lstm<B: Backend> {
    input_gate: Linear<B>,
    forget_gate: Linear<B>,
    cell_gate: Linear<B>,
    output_gate: Linear<B>,
}
```

#### Forward Pass

```rust
impl<B: Backend> Lstm<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let (batch_size, seq_len, input_size) = input.dims();
        let mut hidden_states = Vec::new();
        let mut cell_states = Vec::new();
        
        // Initialize hidden and cell states
        let mut h = Tensor::zeros([batch_size, self.hidden_size], &input.device());
        let mut c = Tensor::zeros([batch_size, self.hidden_size], &input.device());
        
        for t in 0..seq_len {
            let x_t = input.slice([0..batch_size, t..t+1, 0..input_size]);
            
            // Gates computation
            let i_t = self.input_gate.forward(x_t.clone());
            let f_t = self.forget_gate.forward(x_t.clone());
            let c_tilde = self.cell_gate.forward(x_t.clone());
            let o_t = self.output_gate.forward(x_t);
            
            // State updates
            c = f_t * c + i_t * c_tilde;
            h = o_t * c.tanh();
            
            hidden_states.push(h.clone());
            cell_states.push(c.clone());
        }
        
        // Stack all hidden states
        Tensor::stack(hidden_states, 1)
    }
}
```

### GRU (Gated Recurrent Unit)

#### Architecture Overview

```rust
pub struct GruModel<B: Backend> {
    gru: Gru<B>,
    dropout: Dropout,
    linear: Linear<B>,
}
```

#### Cell Implementation

```rust
pub struct Gru<B: Backend> {
    update_gate: Linear<B>,
    reset_gate: Linear<B>,
    candidate_gate: Linear<B>,
}
```

#### Forward Pass

```rust
impl<B: Backend> Gru<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let (batch_size, seq_len, input_size) = input.dims();
        let mut hidden_states = Vec::new();
        
        // Initialize hidden state
        let mut h = Tensor::zeros([batch_size, self.hidden_size], &input.device());
        
        for t in 0..seq_len {
            let x_t = input.slice([0..batch_size, t..t+1, 0..input_size]);
            
            // Gates computation
            let z_t = self.update_gate.forward(x_t.clone());
            let r_t = self.reset_gate.forward(x_t.clone());
            let h_tilde = self.candidate_gate.forward(x_t);
            
            // State update
            h = z_t * h + (1.0 - z_t) * h_tilde;
            
            hidden_states.push(h.clone());
        }
        
        // Stack all hidden states
        Tensor::stack(hidden_states, 1)
    }
}
```

### CNN-LSTM (Convolutional Neural Network + LSTM)

#### Architecture Overview

```rust
pub struct CnnLstmModel<B: Backend> {
    conv_layers: Vec<Conv2d<B>>,
    lstm: Lstm<B>,
    dropout: Dropout,
    linear: Linear<B>,
}
```

#### Implementation

```rust
impl<B: Backend> CnnLstmModel<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        // CNN feature extraction
        let mut x = input;
        for conv in &self.conv_layers {
            x = conv.forward(x);
            x = x.relu();
        }
        
        // Reshape for LSTM
        let (batch, channels, height, width) = x.dims();
        let x = x.reshape([batch, height, channels * width]);
        
        // LSTM processing
        let lstm_out = self.lstm.forward(x);
        
        // Final prediction
        let output = self.linear.forward(lstm_out);
        self.dropout.forward(output)
    }
}
```

## 📊 Data Processing Pipeline

### Feature Engineering

#### Technical Indicators

```rust
pub const TECHNICAL_INDICATORS: [&str; 12] = [
    "close",      // Closing price
    "volume",     // Trading volume
    "sma_20",     // 20-period Simple Moving Average
    "sma_50",     // 50-period Simple Moving Average
    "ema_20",     // 20-period Exponential Moving Average
    "rsi_14",     // 14-period Relative Strength Index
    "macd",       // MACD line
    "macd_signal", // MACD signal line
    "bb_middle",  // Bollinger Bands middle line
    "atr_14",     // 14-period Average True Range
    "returns",    // Price returns
    "price_range", // High-Low range
];
```

#### Extended Indicators

```rust
pub const EXTENDED_INDICATORS: [&str; 23] = [
    // Original indicators...
    "bb_b",           // Bollinger Bands %B
    "gk_volatility",  // Garman-Klass volatility
    // Time-based features
    "hour_sin",       // Hour sine encoding
    "hour_cos",       // Hour cosine encoding
    "day_of_week_sin", // Day of week sine encoding
    "day_of_week_cos", // Day of week cosine encoding
    // Lag features
    "close_lag_5",    // 5-period lag
    "close_lag_15",   // 15-period lag
    "close_lag_30",   // 30-period lag
    "returns_5min",   // 5-minute returns
    "volatility_15min", // 15-minute volatility
];
```

#### Implementation Example

```rust
pub fn add_technical_indicators(df: &mut DataFrame) -> Result<(), PolarsError> {
    // Simple Moving Average
    let sma_20 = df.column("close")?.rolling_mean(RollingOptions {
        window_size: 20,
        min_periods: 1,
        center: false,
    })?;
    df.with_column(sma_20.alias("sma_20"))?;
    
    // RSI calculation
    let rsi_14 = calculate_rsi(df.column("close")?, 14)?;
    df.with_column(rsi_14.alias("rsi_14"))?;
    
    // MACD calculation
    let (macd, signal) = calculate_macd(df.column("close")?, 12, 26, 9)?;
    df.with_column(macd.alias("macd"))?;
    df.with_column(signal.alias("macd_signal"))?;
    
    Ok(())
}
```

### Data Preprocessing

#### Normalization

```rust
pub fn normalize_data(data: &mut DataFrame) -> Result<(), PolarsError> {
    let numeric_columns = data.select_dtypes(&[DataType::Float64, DataType::Float32])?;
    
    for col in numeric_columns.get_column_names() {
        let series = data.column(col)?;
        let mean = series.mean().unwrap_or(0.0);
        let std = series.std().unwrap_or(1.0);
        
        let normalized = (series - mean) / std;
        *data = data.with_column(normalized.alias(col))?;
    }
    
    Ok(())
}
```

#### Sequence Generation

```rust
pub fn create_sequences(
    data: &DataFrame,
    sequence_length: usize,
    target_column: &str,
) -> Result<(Tensor<B, 3>, Tensor<B, 2>), Box<dyn std::error::Error>> {
    let n_samples = data.height() - sequence_length;
    let n_features = data.width() - 1; // Exclude target column
    
    let mut sequences = Vec::new();
    let mut targets = Vec::new();
    
    for i in 0..n_samples {
        let sequence_data = data.slice(i, sequence_length);
        let target_data = data.slice(i + sequence_length, 1);
        
        // Convert to tensors
        let sequence_tensor = dataframe_to_tensor(&sequence_data)?;
        let target_tensor = dataframe_to_tensor(&target_data)?;
        
        sequences.push(sequence_tensor);
        targets.push(target_tensor);
    }
    
    Ok((
        Tensor::stack(sequences, 0),
        Tensor::stack(targets, 0),
    ))
}
```

## 🎯 Training Configuration

### Default Parameters

```rust
// Model parameters
pub const SEQUENCE_LENGTH: usize = 10;
pub const LSTM_TRAINING_DAYS: i64 = 200;
pub const VALIDATION_SPLIT_RATIO: f64 = 0.2;
pub const DEFAULT_DROPOUT: f64 = 0.15;
pub const L2_REGULARIZATION: f64 = 0.01;
```

### Training Configuration

```rust
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub test_split: f64,
    pub dropout: f64,
    pub patience: usize,
    pub min_delta: f64,
    pub use_huber_loss: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            test_split: 0.2,
            dropout: 0.15,
            patience: 5,
            min_delta: 0.001,
            use_huber_loss: true,
        }
    }
}
```

### Loss Functions

#### Mean Squared Error (MSE)

```rust
pub fn mse_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 0> {
    let diff = predictions - targets;
    (diff * diff).mean()
}
```

#### Huber Loss

```rust
pub fn huber_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    delta: f32,
) -> Tensor<B, 0> {
    let diff = predictions - targets;
    let abs_diff = diff.abs();
    
    let quadratic = (diff * diff) * 0.5;
    let linear = abs_diff * delta - 0.5 * delta * delta;
    
    let mask = abs_diff.less_equal(delta);
    (mask * quadratic + (!mask) * linear).mean()
}
```

## 📈 Performance Metrics

### Evaluation Metrics

```rust
pub fn calculate_metrics<B: Backend>(
    predictions: &Tensor<B, 2>,
    targets: &Tensor<B, 2>,
) -> Metrics {
    let mse = mse_loss(predictions.clone(), targets.clone());
    let mae = mae_loss(predictions.clone(), targets.clone());
    let mape = mape_loss(predictions.clone(), targets.clone());
    
    Metrics {
        rmse: mse.sqrt().into_scalar(),
        mae: mae.into_scalar(),
        mape: mape.into_scalar(),
    }
}

pub struct Metrics {
    pub rmse: f32,
    pub mae: f32,
    pub mape: f32,
}
```

### Model Comparison

```rust
pub fn compare_models(
    lstm_predictions: &[f32],
    gru_predictions: &[f32],
    actual_values: &[f32],
) -> ModelComparison {
    let lstm_rmse = calculate_rmse(lstm_predictions, actual_values);
    let gru_rmse = calculate_rmse(gru_predictions, actual_values);
    
    ModelComparison {
        lstm_rmse,
        gru_rmse,
        better_model: if lstm_rmse < gru_rmse { "LSTM" } else { "GRU" },
        improvement: ((gru_rmse - lstm_rmse) / gru_rmse * 100.0).abs(),
    }
}
```

## 💾 Model Serialization

### Save Model

```rust
pub fn save_model<B: Backend>(
    model: &LstmModel<B>,
    metadata: &ModelMetadata,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create directory if it doesn't exist
    std::fs::create_dir_all(path.parent().unwrap())?;
    
    // Save model weights
    let model_bytes = bincode::serialize(&model)?;
    std::fs::write(path, model_bytes)?;
    
    // Save metadata
    let metadata_path = path.with_extension("json");
    let metadata_json = serde_json::to_string_pretty(metadata)?;
    std::fs::write(metadata_path, metadata_json)?;
    
    Ok(())
}
```

### Load Model

```rust
pub fn load_model<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<(LstmModel<B>, ModelMetadata), Box<dyn std::error::Error>> {
    // Load model weights
    let model_bytes = std::fs::read(path)?;
    let model: LstmModel<B> = bincode::deserialize(&model_bytes)?;
    
    // Load metadata
    let metadata_path = path.with_extension("json");
    let metadata_json = std::fs::read_to_string(metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_json)?;
    
    Ok((model, metadata))
}
```

## 🔧 Configuration Management

### Environment Variables

```rust
pub struct Config {
    pub model_path: String,
    pub data_path: String,
    pub log_level: String,
    pub device: String,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            model_path: env::var("MODEL_PATH").unwrap_or_else(|_| "models".to_string()),
            data_path: env::var("DATA_PATH").unwrap_or_else(|_| "examples/csv".to_string()),
            log_level: env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
            device: env::var("DEVICE").unwrap_or_else(|_| "cpu".to_string()),
        }
    }
}
```

### Model Metadata

```rust
#[derive(Serialize, Deserialize, Clone)]
pub struct ModelMetadata {
    pub version: String,
    pub created_at: DateTime<Utc>,
    pub model_type: String,
    pub ticker: String,
    pub sequence_length: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub training_days: i64,
    pub validation_split: f64,
    pub final_loss: f32,
    pub final_rmse: f32,
}
```

## 🚀 Performance Optimization

### Memory Management

```rust
// Use smaller batch sizes for memory-constrained environments
pub const MEMORY_EFFICIENT_BATCH_SIZE: usize = 16;

// Gradient accumulation for effective larger batch sizes
pub fn train_with_gradient_accumulation<B: Backend>(
    model: &mut LstmModel<B>,
    data: &DataLoader<B>,
    config: &TrainingConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let accumulation_steps = 4;
    let effective_batch_size = config.batch_size * accumulation_steps;
    
    for epoch in 0..config.epochs {
        let mut accumulated_gradients = None;
        
        for (batch_idx, (inputs, targets)) in data.enumerate() {
            let loss = model.forward(inputs, targets);
            let gradients = loss.backward();
            
            // Accumulate gradients
            accumulated_gradients = match accumulated_gradients {
                Some(acc) => Some(acc + gradients),
                None => Some(gradients),
            };
            
            // Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 {
                if let Some(grads) = accumulated_gradients.take() {
                    model.update_weights(grads / accumulation_steps as f32);
                }
            }
        }
    }
    
    Ok(())
}
```

### Parallel Processing

```rust
// Use Rayon for parallel data processing
use rayon::prelude::*;

pub fn parallel_feature_engineering(df: &DataFrame) -> Result<DataFrame, PolarsError> {
    let columns: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
    
    let processed_columns: Vec<Series> = columns
        .par_iter()
        .map(|col_name| {
            let series = df.column(col_name)?;
            process_column(series)
        })
        .collect();
    
    DataFrame::new(processed_columns)
}
```

## 📊 Benchmarking

### Performance Benchmarks

```rust
#[cfg(test)]
mod benchmarks {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_lstm_forward(b: &mut Bencher) {
        let device = NdArrayDevice::default();
        let model = LstmModel::new(&device, 10, 64, 2);
        let input = Tensor::random([32, 10, 10], &device);
        
        b.iter(|| {
            model.forward(input.clone());
        });
    }

    #[bench]
    fn bench_gru_forward(b: &mut Bencher) {
        let device = NdArrayDevice::default();
        let model = GruModel::new(&device, 10, 64, 2);
        let input = Tensor::random([32, 10, 10], &device);
        
        b.iter(|| {
            model.forward(input.clone());
        });
    }
}
```

### Memory Usage Monitoring

```rust
pub fn monitor_memory_usage() {
    let memory_info = sysinfo::System::new_all();
    let used_memory = memory_info.used_memory();
    let total_memory = memory_info.total_memory();
    
    log::info!(
        "Memory usage: {:.2} GB / {:.2} GB ({:.1}%)",
        used_memory as f64 / 1024.0 / 1024.0 / 1024.0,
        total_memory as f64 / 1024.0 / 1024.0 / 1024.0,
        (used_memory as f64 / total_memory as f64) * 100.0
    );
}
```

## 🔍 Error Handling

### Custom Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum ClawFoxyVisionError {
    #[error("Data loading error: {0}")]
    DataLoadingError(#[from] Box<dyn std::error::Error>),
    
    #[error("Model training error: {0}")]
    TrainingError(String),
    
    #[error("Prediction error: {0}")]
    PredictionError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),
    
    #[error("File I/O error: {0}")]
    IoError(#[from] std::io::Error),
}
```

### Error Recovery

```rust
pub fn robust_training<B: Backend>(
    model: &mut LstmModel<B>,
    data: &DataLoader<B>,
    config: &TrainingConfig,
) -> Result<(), ClawFoxyVisionError> {
    let mut retry_count = 0;
    const MAX_RETRIES: usize = 3;
    
    loop {
        match train_model(model, data, config) {
            Ok(()) => return Ok(()),
            Err(e) => {
                retry_count += 1;
                if retry_count >= MAX_RETRIES {
                    return Err(ClawFoxyVisionError::TrainingError(
                        format!("Training failed after {} retries: {}", MAX_RETRIES, e)
                    ));
                }
                
                log::warn!("Training failed, retrying ({}/{}): {}", retry_count, MAX_RETRIES, e);
                
                // Reset model state for retry
                model.reset_parameters();
            }
        }
    }
}
```

## 📚 API Reference

### Public API

```rust
// Main library exports
pub mod daily;
pub mod minute;
pub mod util;

// Re-export commonly used types
pub use daily::{lstm::*, gru::*};
pub use minute::{lstm::*, gru::*, cnnlstm::*};
pub use util::{file_utils, feature_engineering, model_utils};

// Main entry point
pub fn train_and_predict(
    ticker: &str,
    model_type: &str,
    data_path: &str,
) -> Result<PredictionResult, ClawFoxyVisionError> {
    // Implementation
}
```

### Type Definitions

```rust
pub type BurnBackend = Autodiff<NdArray<f32>>;
pub type Device = NdArrayDevice;

pub struct PredictionResult {
    pub predictions: Vec<f32>,
    pub confidence: Vec<f32>,
    pub metrics: Metrics,
    pub model_path: PathBuf,
}
```

---

**This technical reference provides comprehensive details about ClawFoxyVision's implementation. For usage examples, see the [User Guide](./user-guide.md).** 🔮 