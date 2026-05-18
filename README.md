# ML_Benchmark
This project is an Edge AI Systems Lab that explores how machine learning models behave across different hardware environments and optimization levels. It consists of three models built in PyTorch: a sensor-based anomaly detection model 
that identifies abnormal patterns in time-series data (simulating edge sensor monitoring), a lightweight vision classifier that detects image categories using a small CNN (representing perception tasks like those in robotics or 
autonomous systems), and a regression forecasting model that predicts future values from sequential data (capturing real-world prediction tasks like energy or system load forecasting). All models are exported and deployed through ONNX 
Runtime, then evaluated across three environments: CPU (baseline general-purpose computing), GPU (accelerated high-performance inference), and an edge device such as a Raspberry Pi (resource-constrained real-world deployment). The goal 
of the project is to analyze the tradeoffs between accuracy, latency, memory usage, and computational efficiency across hardware platforms, demonstrating how modern AI systems must be optimized not just for intelligence, but for 
practical deployment under real-world hardware constraints.
