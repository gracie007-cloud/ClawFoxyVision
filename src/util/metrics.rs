use once_cell::sync::Lazy;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::time::Instant;

static PROM_HANDLE: Lazy<PrometheusHandle> = Lazy::new(|| {
    PrometheusBuilder::new()
        .with_http_listener(([0, 0, 0, 0], 9100))
        .install()
        .expect("failed to install Prometheus recorder")
});

/// Initialise metrics exporter (called once at application start).
pub fn init_metrics() {
    Lazy::force(&PROM_HANDLE);
    log::info!("Prometheus metrics exporter running on 0.0.0.0:9100");
}

/// Utility to record execution duration in milliseconds.
#[macro_export]
macro_rules! record_timed {
    ($name:expr, $block:block) => {{
        let _span_start = std::time::Instant::now();
        let result = { $block };
        let elapsed_ms = _span_start.elapsed().as_millis() as u64;
        metrics::histogram!(concat!($name, "_ms"), elapsed_ms as f64);
        result
    }};
} 