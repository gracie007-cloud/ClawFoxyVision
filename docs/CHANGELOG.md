# Changelog

All notable changes to ClawFoxyVision will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation suite
- User guide for end users
- Developer guide for contributors
- Technical reference for implementation details

## [0.2.0] - 2024-01-XX

### Added
- CNN-LSTM model implementation for minute-level data
- Enhanced feature engineering with extended technical indicators
- Time-based features (hour, day of week encoding)
- Lag features for improved pattern recognition
- Model versioning and metadata tracking
- Improved error handling and recovery mechanisms
- Memory-efficient training with gradient accumulation
- Parallel data processing with Rayon
- Comprehensive test suite
- Performance benchmarking tools

### Changed
- Updated to Burn framework 0.17.0
- Improved model architecture with better regularization
- Enhanced data preprocessing pipeline
- Optimized memory usage for large datasets
- Better model serialization format

### Fixed
- Memory issues with large training datasets
- Model loading compatibility issues
- Data preprocessing edge cases
- Training stability improvements

## [0.1.0] - 2024-01-XX

### Added
- Initial release of ClawFoxyVision
- LSTM model implementation for minute-level data
- GRU model implementation for minute-level data
- Basic feature engineering with technical indicators
- Data preprocessing and normalization
- Model training and evaluation pipeline
- Prediction generation capabilities
- Model serialization and loading
- Command-line interface
- Sample data and examples

### Features
- Support for CSV and Parquet data formats
- OHLC (Open, High, Low, Close) data processing
- Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR
- Configurable model hyperparameters
- Training/validation split functionality
- Performance metrics calculation (RMSE, MAE, MAPE)
- Model comparison capabilities

---

## Version History

- **0.2.0**: Enhanced models, improved performance, comprehensive testing
- **0.1.0**: Initial release with basic LSTM and GRU implementations

## Contributing

When contributing to this project, please update this changelog with your changes following the format above. 