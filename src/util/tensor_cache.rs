use std::fs;
use std::path::{PathBuf};
use burn::tensor::{Tensor, backend::Backend};
use sha2::{Sha256, Digest};
use anyhow::{Result, Context};

/// Return (features, targets) tensors either loaded from cache or computed via callback
pub fn cache_or_compute<B, F>(key_inputs: &str, compute: F) -> Result<(Tensor<B,3>, Tensor<B,2>)>
where
    B: Backend,
    F: FnOnce() -> Result<(Tensor<B,3>, Tensor<B,2>)>,
{
    let cache_dir = std::env::var("CACHE_DIR").unwrap_or_else(|_| "cache/tensors".to_string());
    fs::create_dir_all(&cache_dir)?;

    let mut hasher = Sha256::new();
    hasher.update(key_inputs.as_bytes());
    let hash = format!("{:x}", hasher.finalize());
    let feat_path = PathBuf::from(&cache_dir).join(format!("{}_x.bin", hash));
    let tgt_path = PathBuf::from(&cache_dir).join(format!("{}_y.bin", hash));

    if feat_path.exists() && tgt_path.exists() {
        let feat_bytes = fs::read(&feat_path).context("read features cache")?;
        let tgt_bytes = fs::read(&tgt_path).context("read targets cache")?;
        let features: Tensor<B,3> = bincode::deserialize(&feat_bytes)?;
        let targets: Tensor<B,2> = bincode::deserialize(&tgt_bytes)?;
        return Ok((features, targets));
    }

    let (features, targets) = compute()?;

    fs::write(&feat_path, bincode::serialize(&features)?)?;
    fs::write(&tgt_path, bincode::serialize(&targets)?)?;

    Ok((features, targets))
}

/// Attempt to load tensors; returns Some if present.
pub fn load<B: Backend>(key_inputs: &str) -> Result<Option<(Tensor<B,3>, Tensor<B,2>)>> {
    let cache_dir = std::env::var("CACHE_DIR").unwrap_or_else(|_| "cache/tensors".to_string());
    fs::create_dir_all(&cache_dir)?;
    let mut hasher = Sha256::new();
    hasher.update(key_inputs.as_bytes());
    let hash = format!("{:x}", hasher.finalize());
    let feat_path = PathBuf::from(&cache_dir).join(format!("{}_x.bin", hash));
    let tgt_path = PathBuf::from(&cache_dir).join(format!("{}_y.bin", hash));
    if feat_path.exists() && tgt_path.exists() {
        let feat_bytes = fs::read(&feat_path)?;
        let tgt_bytes = fs::read(&tgt_path)?;
        let features: Tensor<B,3> = bincode::deserialize(&feat_bytes)?;
        let targets: Tensor<B,2> = bincode::deserialize(&tgt_bytes)?;
        return Ok(Some((features, targets)));
    }
    Ok(None)
}

pub fn save<B: Backend>(key_inputs: &str, tensors: &(Tensor<B,3>, Tensor<B,2>)) -> Result<()> {
    let cache_dir = std::env::var("CACHE_DIR").unwrap_or_else(|_| "cache/tensors".to_string());
    fs::create_dir_all(&cache_dir)?;
    let mut hasher = Sha256::new();
    hasher.update(key_inputs.as_bytes());
    let hash = format!("{:x}", hasher.finalize());
    let feat_path = PathBuf::from(&cache_dir).join(format!("{}_x.bin", hash));
    let tgt_path = PathBuf::from(&cache_dir).join(format!("{}_y.bin", hash));
    let (ref features, ref targets) = *tensors;
    fs::write(&feat_path, bincode::serialize(features)?)?;
    fs::write(&tgt_path, bincode::serialize(targets)?)?;
    Ok(())
} 