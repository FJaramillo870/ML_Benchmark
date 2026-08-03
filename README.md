# ML_Benchmark
This project is an Edge AI Systems Lab that explores how machine learning models behave across different hardware environments and optimization levels. It consists of three models built in PyTorch: a sensor-based anomaly detection model 
that identifies abnormal patterns in time-series data (simulating edge sensor monitoring), a lightweight vision classifier that detects image categories using a small CNN (representing perception tasks like those in robotics or 
autonomous systems), and a regression forecasting model that predicts future values from sequential data (capturing real-world prediction tasks like energy or system load forecasting). All models are exported and deployed through ONNX 
Runtime, then evaluated across three environments: CPU (baseline general-purpose computing), GPU (accelerated high-performance inference), and an edge device such as a Raspberry Pi (resource-constrained real-world deployment). The goal 
of the project is to analyze the tradeoffs between accuracy, latency, memory usage, and computational efficiency across hardware platforms, demonstrating how modern AI systems must be optimized not just for intelligence, but for 
practical deployment under real-world hardware constraints.

### Baseline Performance Metrics (CPU)
| Model | Architecture | Target Application | Primary Metric | CPU Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1** | 1D CNN / Autoencoder | Sensor Anomaly Detection | Threshold: 0.5228 MSE | ~0.05 ms (Batch 1) |
| **Model 2** | 2D CNN | Vision Classification | 71.07% Accuracy | ~0.08 ms (Batch 1) |
| **Model 3** | LSTM | Energy Load Forecasting | 0.0072 MSE | ~0.86 ms (Batch 32)* |

*\*Note: Model 3 latency recorded at Batch Size 32. Models 1 & 2 recorded at Batch Size 1.*
