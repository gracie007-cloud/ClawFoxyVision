# ClawFoxyVision Developer Guide

Welcome, developers! This guide will help you understand the ClawFoxyVision codebase, set up your development environment, and contribute to the project.

## 🛠️ Development Setup

### Prerequisites

- **Rust 1.65 or higher** - [Install Rust](https://rustup.rs/)
- **Git** - [Install Git](https://git-scm.com/)
- **A code editor** - VS Code, IntelliJ IDEA, or your preferred editor
- **Basic knowledge of Rust and machine learning concepts**

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rustic-ml/ClawFoxyVision
   cd ClawFoxyVision
   ```

2. **Install dependencies:**
   ```bash
   cargo build
   ```

3. **Run tests to verify setup:**
   ```bash
   cargo test
   ```

4. **Install development tools (optional but recommended):**
   ```bash
   # Rust formatter
   rustup component add rustfmt
   
   # Clippy linter
   rustup component add clippy
   
   # Code coverage (optional)
   cargo install cargo-tarpaulin
   ```

## 🏗️ Project Architecture

### Directory Structure

```
ClawFoxyVision/
├── src/
│   ├── main.rs              # Application entry point
│   ├── lib.rs               # Library exports
│   ├── constants.rs         # Global constants and configuration
│   ├── daily/               # Daily data processing modules
│   │   ├── lstm/           # LSTM implementation for daily data
│   │   └── gru/            # GRU implementation for daily data
│   ├── minute/              # Minute data processing modules
│   │   ├── lstm/           # LSTM implementation for minute data
│   │   ├── gru/            # GRU implementation for minute data
│   │   └── cnnlstm/        # CNN-LSTM implementation
│   ├── util/                # Utility modules
│   │   ├── file_utils.rs   # File I/O operations
│   │   ├── feature_engineering.rs # Technical indicators
│   │   ├── model_utils.rs  # Model management
│   │   └── pre_processor.rs # Data preprocessing
│   └── test/                # Test modules
├── examples/                # Example code and sample data
├── docs/                    # Documentation
└── Cargo.toml              # Project dependencies
```

### Core Modules

#### 1. **Data Processing (`src/util/`)**
- `file_utils.rs` - Handles CSV/Parquet file reading and writing
- `feature_engineering.rs` - Calculates technical indicators
- `pre_processor.rs` - Data normalization and preprocessing

#### 2. **Model Implementations**
Each model type follows a consistent 6-step pattern:

1. **Tensor Preparation** (`step_1_tensor_preparation.rs`)
   - Data loading and preprocessing
   - Feature engineering
   - Sequence creation

2. **Cell Implementation** (`step_2_*_cell.rs`)
   - Core neural network cell (LSTM/GRU/CNN)
   - Forward pass logic

3. **Model Architecture** (`step_3_*_model_arch.rs`)
   - Complete model structure
   - Layer definitions

4. **Training** (`step_4_train_model.rs`)
   - Training loop
   - Loss calculation
   - Optimization

5. **Prediction** (`step_5_prediction.rs`)
   - Inference logic
   - Prediction generation

6. **Serialization** (`step_6_model_serialization.rs`)
   - Model saving/loading
   - Metadata management

#### 3. **Configuration (`src/constants.rs`)**
- Model hyperparameters
- Technical indicator definitions
- File paths and constants

## 🔧 Development Workflow

### Code Style Guidelines

#### Rust Conventions
- **Formatting:** Use `rustfmt` with project settings
- **Linting:** Use `clippy` for code quality
- **Documentation:** Document all public APIs with rustdoc

#### Naming Conventions
- **Modules:** `snake_case`
- **Functions:** `snake_case`
- **Structs:** `PascalCase`
- **Traits:** `PascalCase`
- **Constants:** `SCREAMING_SNAKE_CASE`

#### Code Organization
```rust
// 1. External imports
use burn_autodiff::Autodiff;
use polars::prelude::*;

// 2. Internal imports
use crate::util::file_utils;

// 3. Constants
const DEFAULT_BATCH_SIZE: usize = 32;

// 4. Structs and types
pub struct ModelConfig {
    // ...
}

// 5. Implementation blocks
impl ModelConfig {
    // ...
}

// 6. Public functions
pub fn train_model() -> Result<(), Error> {
    // ...
}
```

### Testing Strategy

#### Unit Tests
- Test individual functions and modules
- Use `#[cfg(test)]` modules
- Mock external dependencies

#### Integration Tests
- Test complete workflows
- Use sample data from `examples/csv/`
- Verify model training and prediction

#### Example Test Structure
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_engineering() {
        // Test technical indicator calculation
    }

    #[test]
    fn test_model_training() {
        // Test complete training workflow
    }
}
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test module
cargo test test_feature_engineering

# Run tests with output
cargo test -- --nocapture

# Run tests with coverage
cargo tarpaulin
```

## 🚀 Adding New Features

### 1. **Adding a New Model Type**

1. **Create the module structure:**
   ```bash
   mkdir -p src/minute/newmodel
   touch src/minute/newmodel/mod.rs
   touch src/minute/newmodel/step_1_tensor_preparation.rs
   # ... create all 6 step files
   ```

2. **Implement the 6-step pattern:**
   - Follow the existing LSTM/GRU implementations
   - Use the same interfaces and patterns
   - Add appropriate tests

3. **Update main.rs:**
   - Add the new model type to the command-line interface
   - Implement the training and evaluation logic

### 2. **Adding New Technical Indicators**

1. **Edit `src/util/feature_engineering.rs`:**
   ```rust
   pub fn calculate_new_indicator(df: &mut DataFrame) -> Result<(), PolarsError> {
       // Implementation
   }
   ```

2. **Update `src/constants.rs`:**
   ```rust
   pub const TECHNICAL_INDICATORS: [&str; 13] = [
       // ... existing indicators
       "new_indicator",
   ];
   ```

3. **Add tests:**
   ```rust
   #[test]
   fn test_new_indicator() {
       // Test the new indicator
   }
   ```

### 3. **Adding New Data Sources**

1. **Extend `src/util/file_utils.rs`:**
   ```rust
   pub fn read_new_format(file_path: &str) -> Result<DataFrame, Box<dyn std::error::Error>> {
       // Implementation
   }
   ```

2. **Update the main data loading function:**
   - Add format detection logic
   - Handle the new format appropriately

## 🔍 Debugging and Profiling

### Debugging Tips

1. **Use logging:**
   ```rust
   use log::{info, warn, error};

   info!("Training model with {} samples", data.len());
   warn!("High memory usage detected");
   error!("Failed to load model: {}", e);
   ```

2. **Enable debug logging:**
   ```bash
   RUST_LOG=debug cargo run -- AAPL lstm
   ```

3. **Use `dbg!` macro for quick debugging:**
   ```rust
   let result = some_function();
   dbg!(&result);
   ```

### Performance Profiling

1. **Use `cargo bench` for benchmarking:**
   ```rust
   #[cfg(test)]
   mod benches {
     use super::*;
     use test::Bencher;

     #[bench]
     fn bench_model_training(b: &mut Bencher) {
         b.iter(|| {
             // Benchmark code
         });
     }
   }
   ```

2. **Memory profiling with `cargo install flamegraph`:**
   ```bash
   cargo flamegraph --bin ClawFoxyVision -- AAPL lstm
   ```

## 📦 Building and Distribution

### Building for Release

```bash
# Optimized release build
cargo build --release

# Check binary size
ls -lh target/release/ClawFoxyVision
```

### Cross-Platform Building

```bash
# Install cross-compilation targets
rustup target add x86_64-unknown-linux-gnu
rustup target add x86_64-pc-windows-gnu
rustup target add x86_64-apple-darwin

# Build for different platforms
cargo build --release --target x86_64-unknown-linux-gnu
cargo build --release --target x86_64-pc-windows-gnu
cargo build --release --target x86_64-apple-darwin
```

### Documentation Generation

```bash
# Generate API documentation
cargo doc --no-deps --open

# Generate and check documentation
cargo doc --document-private-items
```

## 🤝 Contributing Guidelines

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/new-feature
   ```

3. **Make your changes:**
   - Follow the coding style guidelines
   - Add appropriate tests
   - Update documentation

4. **Run quality checks:**
   ```bash
   cargo fmt
   cargo clippy
   cargo test
   ```

5. **Commit your changes:**
   ```bash
   git commit -m "feat: add new technical indicator"
   ```

6. **Push and create a pull request**

### Commit Message Format

Use conventional commit format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions or changes
- `chore:` Maintenance tasks

### Code Review Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] No performance regressions
- [ ] Error handling is appropriate
- [ ] Security considerations addressed

## 🐛 Common Issues and Solutions

### Build Issues

**"Cannot find burn crate"**
```bash
# Ensure you have the correct Rust version
rustup update
cargo clean
cargo build
```

**"Out of memory during compilation"**
```bash
# Increase memory limit
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$ORIGIN"
cargo build --release
```

### Runtime Issues

**"Model loading fails"**
- Check file permissions
- Verify model file integrity
- Ensure correct model version

**"Poor prediction accuracy"**
- Check data quality
- Verify feature engineering
- Adjust hyperparameters

## 📚 Learning Resources

### Rust Resources
- [The Rust Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

### Machine Learning Resources
- [Burn Framework Documentation](https://github.com/tracel-ai/burn)
- [Time Series Analysis](https://otexts.com/fpp3/)
- [Deep Learning for Time Series](https://www.oreilly.com/library/view/deep-learning-for/9781492044459/)

### Project-Specific Resources
- [Examples Directory](../examples/) - Working code examples
- [Technical Reference](./technical-reference.md) - Detailed implementation docs
- [GitHub Issues](https://github.com/rustic-ml/ClawFoxyVision/issues) - Known issues and discussions

## 🎯 Next Steps

1. **Explore the codebase** - Start with `src/main.rs` and follow the execution flow
2. **Run examples** - Try the examples in the `examples/` directory
3. **Pick an issue** - Look for "good first issue" labels on GitHub
4. **Join discussions** - Participate in GitHub discussions and issues
5. **Contribute** - Submit your first pull request!

---

**Happy coding! Let's make ClawFoxyVision even better together.** 🚀 