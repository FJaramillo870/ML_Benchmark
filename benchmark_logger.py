import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import csv
from datetime import datetime

# Place the results inside your existing 'benchmarks' folder
CSV_PATH = os.path.join("benchmarks", "benchmark_results.csv")


def log_benchmark(model_id, architecture, precision, metric_name, metric_value, latency_ms, batch_size, model_size_mb):
    """
    Appends a standardized benchmark result entry to the benchmarks CSV file.
    """
    os.makedirs("benchmarks", exist_ok=True)
    file_exists = os.path.isfile(CSV_PATH)

    headers = [
        "Timestamp",
        "Model ID",
        "Architecture",
        "Precision",
        "Metric Name",
        "Metric Value",
        "Latency (ms)",
        "Batch Size",
        "Model Size (MB)"
    ]

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_id,
        architecture,
        precision,
        metric_name,
        round(float(metric_value), 6),
        round(float(latency_ms), 4),
        batch_size,
        round(float(model_size_mb), 4)
    ]

    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)  # Write headers if the file is new
        writer.writerow(row)

    print(f"\n[AUTO-LOGGER] Successfully appended benchmark results to: {CSV_PATH}")


if __name__ == "__main__":
    print("Testing Benchmark Logger...")
    # Test entry to verify file creation
    log_benchmark("Model 0", "Test Architecture", "float32", "Test Accuracy", 99.9, 0.01, 1, 0.05)