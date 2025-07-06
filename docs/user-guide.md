# ClawFoxyVision User Guide

Welcome to ClawFoxyVision! This guide will help you get started with using our advanced financial time series forecasting library for price prediction.

## 🚀 Quick Start

### Prerequisites

- **Rust 1.65 or higher** - [Install Rust](https://rustup.rs/)
- **Stock data in CSV format** with OHLC (Open, High, Low, Close) values
- **Basic command-line knowledge**

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rustic-ml/ClawFoxyVision
   cd ClawFoxyVision
   ```

2. **Build the project:**
   ```bash
   cargo build --release
   ```

## 📊 Data Requirements

### Supported Data Formats

ClawFoxyVision supports both CSV and Parquet files with the following required columns:

- `symbol` - Stock ticker symbol
- `datetime` - Timestamp in ISO format
- `open` - Opening price
- `high` - Highest price
- `low` - Lowest price
- `close` - Closing price
- `volume` - Trading volume

### Optional Columns

- `adjusted_close` - Adjusted closing price

### Example Data Structure

```csv
symbol,datetime,open,high,low,close,volume
AAPL,2024-01-01T09:30:00,150.00,151.50,149.80,150.25,1000000
AAPL,2024-01-01T09:31:00,150.25,150.80,150.10,150.60,950000
```

### Sample Data

The project includes sample data files in the `examples/csv/` directory:
- `AAPL_minute_ohlcv.csv` - Apple minute-level data
- `TSLA_daily_ohlcv.csv` - Tesla daily data
- And many more...

## 🎯 Running Models

### Using the Shell Script (Recommended)

The easiest way to run models is using the provided shell script:

```bash
./run_model.sh [ticker] [model_type]
```

**Examples:**
```bash
# Run LSTM model on Apple stock
./run_model.sh AAPL lstm

# Run GRU model on Tesla stock
./run_model.sh TSLA gru

# Run CNN-LSTM model on Google stock
./run_model.sh GOOGL cnnlstm
```

### Using Cargo Directly

```bash
cargo run --release -- [ticker] [model_type]
```

**Examples:**
```bash
cargo run --release -- AAPL lstm
cargo run --release -- TSLA gru
cargo run --release -- GOOGL cnnlstm
```

## 🧠 Model Types

### LSTM (Long Short-Term Memory)
- **Best for:** Complex patterns and long-term dependencies
- **Use case:** When you need to capture intricate market dynamics
- **Performance:** Generally more accurate but slower training

### GRU (Gated Recurrent Unit)
- **Best for:** Faster training and simpler patterns
- **Use case:** When you need quick results or have limited computational resources
- **Performance:** Faster training, slightly less accurate than LSTM

### CNN-LSTM (Convolutional Neural Network + LSTM)
- **Best for:** Pattern recognition in time series data
- **Use case:** When you want to capture both local and global patterns
- **Performance:** Good balance of speed and accuracy

## 📈 Understanding Results

### Model Output

When you run a model, ClawFoxyVision will:

1. **Load and preprocess** your data
2. **Train the model** on historical data
3. **Generate predictions** for future time periods
4. **Save the trained model** for future use
5. **Display performance metrics**

### Performance Metrics

The model provides several metrics to evaluate performance:

- **RMSE (Root Mean Square Error)** - Lower is better
- **MAE (Mean Absolute Error)** - Lower is better
- **MAPE (Mean Absolute Percentage Error)** - Lower is better

### Output Files

Models are saved in the `models/` directory:
- `{TICKER}_lstm_model` - LSTM model files
- `{TICKER}_gru_model` - GRU model files
- `{TICKER}_cnnlstm_model` - CNN-LSTM model files

## 🔧 Configuration Options

### Model Parameters

You can modify model behavior by editing `src/constants.rs`:

```rust
// Sequence length (how many time steps to look back)
pub const SEQUENCE_LENGTH: usize = 10;

// Training days
pub const LSTM_TRAINING_DAYS: i64 = 200;

// Validation split ratio
pub const VALIDATION_SPLIT_RATIO: f64 = 0.2;

// Dropout rate
pub const DEFAULT_DROPOUT: f64 = 0.15;
```

### Training Configuration

For advanced users, you can modify training parameters in the code:

- `learning_rate` - How fast the model learns
- `batch_size` - Number of samples per training batch
- `epochs` - Number of training iterations
- `dropout` - Regularization to prevent overfitting

## 📊 Examples and Use Cases

### Basic Price Prediction

```bash
# Predict Apple stock prices using LSTM
./run_model.sh AAPL lstm
```

### Model Comparison

```bash
# Compare LSTM and GRU performance
./run_model.sh AAPL lstm
./run_model.sh AAPL gru
```

### Different Timeframes

The library supports both minute-level and daily data:

- **Minute data:** For intraday trading strategies
- **Daily data:** For swing trading and long-term analysis

## 🚨 Troubleshooting

### Common Issues

**1. "File not found" error**
- Ensure your CSV file is in the `examples/csv/` directory
- Check the file naming convention: `{TICKER}_minute_ohlcv.csv`

**2. "Out of memory" error**
- Reduce `LSTM_TRAINING_DAYS` in `src/constants.rs`
- Use a smaller dataset
- Try the GRU model instead of LSTM

**3. "Invalid model type" error**
- Use only: `lstm`, `gru`, or `cnnlstm`
- Check spelling and case sensitivity

**4. Poor prediction accuracy**
- Ensure your data quality is good
- Try different model types
- Adjust sequence length and training parameters
- Check for data leakage or overfitting

### Performance Tips

1. **Use release builds** for better performance:
   ```bash
   cargo run --release -- AAPL lstm
   ```

2. **Monitor system resources** during training

3. **Start with smaller datasets** to test your setup

4. **Use appropriate model types** for your use case

## 📚 Advanced Usage

### Custom Data Sources

To use your own data:

1. Place your CSV file in `examples/csv/`
2. Follow the naming convention: `{TICKER}_minute_ohlcv.csv`
3. Ensure all required columns are present

### Batch Processing

For multiple stocks, you can create a simple script:

```bash
#!/bin/bash
for ticker in AAPL TSLA GOOGL MSFT; do
    echo "Processing $ticker..."
    ./run_model.sh $ticker lstm
done
```

### Integration with Trading Systems

The saved models can be loaded and used for real-time predictions:

```rust
use ClawFoxyVision::util::model_utils;

// Load a trained model
let (model, metadata) = model_utils::load_trained_lstm_model::<BurnBackend>(
    "AAPL", "lstm", "AAPL_lstm_model", &device
)?;
```

## 🤝 Getting Help

If you encounter issues:

1. **Check this guide** for common solutions
2. **Review the examples** in the `examples/` directory
3. **Check the logs** for detailed error messages
4. **Open an issue** on GitHub with:
   - Your command and error message
   - Data format and sample
   - System information

## 📖 Next Steps

- **Explore examples** in the `examples/` directory
- **Read the Technical Reference** for advanced features
- **Check the Developer Guide** if you want to contribute
- **Join the community** on GitHub

---

**Happy forecasting! May Clawy and Foxy guide your financial decisions.** 🔮 