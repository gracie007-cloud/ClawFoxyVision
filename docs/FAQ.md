# Frequently Asked Questions (FAQ)

This document answers common questions about ClawFoxyVision.

## 🚀 Getting Started

### Q: What is ClawFoxyVision?
**A:** ClawFoxyVision is an advanced financial time series forecasting library built in Rust using the Burn deep learning framework. It implements LSTM, GRU, and CNN-LSTM models for stock price prediction.

### Q: What are the system requirements?
**A:** 
- Rust 1.65 or higher
- At least 4GB RAM (8GB recommended)
- CPU with multi-core support
- Linux, macOS, or Windows

### Q: How do I install ClawFoxyVision?
**A:** 
```bash
git clone https://github.com/rustic-ml/ClawFoxyVision
cd ClawFoxyVision
cargo build --release
```

## 📊 Data and Models

### Q: What data formats are supported?
**A:** ClawFoxyVision supports both CSV and Parquet files with OHLC (Open, High, Low, Close) data plus volume.

### Q: What's the difference between LSTM, GRU, and CNN-LSTM?
**A:** 
- **LSTM**: Best for complex patterns, more accurate but slower
- **GRU**: Faster training, good for simpler patterns
- **CNN-LSTM**: Combines pattern recognition with sequence modeling

### Q: How much data do I need?
**A:** Minimum 200 days of data, but more data generally leads to better predictions. The library automatically uses the last 200 days for training.

### Q: Can I use my own data?
**A:** Yes! Place your CSV file in `examples/csv/` with the naming convention `{TICKER}_minute_ohlcv.csv` and ensure it has the required columns.

## 🔧 Usage and Configuration

### Q: How do I run a model?
**A:** 
```bash
./run_model.sh AAPL lstm
# or
cargo run --release -- AAPL lstm
```

### Q: How do I compare models?
**A:** Run different models on the same ticker:
```bash
./run_model.sh AAPL lstm
./run_model.sh AAPL gru
./run_model.sh AAPL cnnlstm
```

### Q: Where are models saved?
**A:** Models are saved in the `models/` directory with names like `AAPL_lstm_model`.

### Q: How do I change model parameters?
**A:** Edit `src/constants.rs` to modify hyperparameters like sequence length, training days, and dropout rate.

## 🐛 Troubleshooting

### Q: I get "File not found" error
**A:** 
- Check that your CSV file is in `examples/csv/`
- Verify the naming convention: `{TICKER}_minute_ohlcv.csv`
- Ensure the file has the required columns

### Q: I get "Out of memory" error
**A:** 
- Reduce `LSTM_TRAINING_DAYS` in `src/constants.rs`
- Use a smaller dataset
- Try the GRU model instead of LSTM
- Close other applications to free memory

### Q: My predictions are inaccurate
**A:** 
- Check data quality and completeness
- Try different model types
- Adjust sequence length and training parameters
- Ensure no data leakage in your preprocessing

### Q: The model is training very slowly
**A:** 
- Use release builds: `cargo run --release`
- Reduce batch size in training configuration
- Use GRU instead of LSTM for faster training
- Consider using a smaller dataset for testing

### Q: I get "Invalid model type" error
**A:** Use only: `lstm`, `gru`, or `cnnlstm`. Check spelling and case sensitivity.

## 🔍 Advanced Usage

### Q: How do I add new technical indicators?
**A:** 
1. Edit `src/util/feature_engineering.rs`
2. Add your indicator calculation function
3. Update `src/constants.rs` to include the new indicator
4. Add tests for the new indicator

### Q: Can I use GPU acceleration?
**A:** Currently, ClawFoxyVision uses CPU with NdArray backend. GPU support may be added in future versions.

### Q: How do I integrate with trading systems?
**A:** 
```rust
use ClawFoxyVision::util::model_utils;

let (model, metadata) = model_utils::load_trained_lstm_model::<BurnBackend>(
    "AAPL", "lstm", "AAPL_lstm_model", &device
)?;
```

### Q: How do I batch process multiple stocks?
**A:** 
```bash
#!/bin/bash
for ticker in AAPL TSLA GOOGL MSFT; do
    echo "Processing $ticker..."
    ./run_model.sh $ticker lstm
done
```

## 📈 Performance and Optimization

### Q: How can I improve prediction accuracy?
**A:** 
- Use more historical data
- Try different model types
- Adjust hyperparameters
- Add more technical indicators
- Ensure data quality

### Q: What do the performance metrics mean?
**A:** 
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)

### Q: How long does training take?
**A:** Training time depends on:
- Dataset size (typically 5-15 minutes for 200 days)
- Model type (GRU is faster than LSTM)
- Hardware specifications
- Batch size and epochs

### Q: How much memory does training use?
**A:** Memory usage depends on:
- Dataset size
- Batch size
- Model complexity
- Typically 2-8GB for standard datasets

## 🤝 Contributing

### Q: How can I contribute?
**A:** 
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Q: What coding standards should I follow?
**A:** 
- Use `rustfmt` for formatting
- Use `clippy` for linting
- Follow Rust naming conventions
- Document public APIs
- Add tests for new features

### Q: How do I run tests?
**A:** 
```bash
cargo test
cargo test -- --nocapture  # See output
cargo tarpaulin  # Coverage
```

## 📚 Resources

### Q: Where can I learn more about the algorithms?
**A:** 
- [Burn Framework Documentation](https://github.com/tracel-ai/burn)
- [Time Series Analysis](https://otexts.com/fpp3/)
- [Deep Learning for Time Series](https://www.oreilly.com/library/view/deep-learning-for/9781492044459/)

### Q: Where can I get help?
**A:** 
1. Check this FAQ
2. Review the documentation
3. Look at examples in the `examples/` directory
4. Open an issue on GitHub
5. Check the troubleshooting sections in the guides

### Q: Is there a community or forum?
**A:** Currently, the main community interaction is through GitHub issues and discussions. Join the conversation there!

---

**Still have questions?** Check the [User Guide](./user-guide.md), [Developer Guide](./developer-guide.md), or [Technical Reference](./technical-reference.md) for more detailed information. 