# Model Evaluation & Benchmarking

### **How it works?**

1. **Download and Evaluate**: Run the model on a standardized dataset (e.g., ImageNet, COCO).  
2. **Measure Performance**:  
   - Use tools like ONNX Runtime, TensorFlow Lite Benchmark Tool, or CoreML `Profiler`.  
   - Record latency on mobile/edge devices (e.g., Raspberry Pi, Jetson Nano).  
3. **Document Results**: Use the table above to document key metrics.  
4. **Submit with PR**: Attach the benchmark results with your PR submission. 


## **Evaluation/Profiling Key Metrics (General Template)**

The goal is to optimize the model performance to meet the HW Platform requirements (Hardware-aware).

**Edge Devices**

| **Model Watch Metrics** | **Description**     |
|-------------------------|---------------------|
|          | Monitors the model's performance and resource usage over time. |
| Model Params Size (MB)         | The size of the model's parameters in megabytes, which impacts memory usage and loading time. |
| Model Size (MB)         | The size of the model file in megabytes, indicating storage requirements. |
| Accuracy (%)            | The percentage of correct predictions made by the model. |
| Latency (ms)            | The time taken to process a single input, measured in milliseconds. |
| Inference Time (ms)     | The total time taken to complete a single inference, including data preprocessing and postprocessing. |
| Throughput (inferences/sec) | The number of inferences the model can perform per second. |
| Memory Footprint (MB)   | The total amount of memory used by the model during execution. |
| Error Rate (%)          | The percentage of incorrect predictions made by the model. |
| Deployment Time (s)     | The time taken to deploy the model onto the device. |

| **Target HW Platform Watch Metrics** | **Description**     |
|--------------------------------------|---------------------|
|             | Monitors the hardware platform's performance and resource usage. |
| Computing Workload (%): CPU, GPU, NPU, TPU, VPU | The percentage of computing resources utilized by the model on different processing units. |
| Computing Speed (MHz)                | The operating frequency of the processing units, measured in megahertz. |
| Energy Consumption (mWh)             | Measures the amount of energy consumed by the model during inference. |
| Temperature (°C)                     | The operating temperature of the device while running the model. |
| Startup Time (ms)                    | The time taken for the model to load and be ready for inference. |
| Network Bandwidth (MB/s)             | The amount of data transferred over the network per second. |
| Model Complexity (FLOPs)             | The number of floating-point operations required for a single inference. |

**MCUs**

| **Model Watch Metrics** | **Description**     |
|-------------------------|---------------------|
|            | Monitors the model's performance and resource usage over time. |
| Model Params Size (MB)         | The size of the model's parameters in megabytes, which impacts memory usage and loading time. |
| Model Size (MB)         | The size of the model file in megabytes, indicating storage requirements. |
| Accuracy (%)            | The percentage of correct predictions made by the model for image classification tasks. |
| Latency (ms)            | The time taken to process a single input for image classification, measured in milliseconds. |
| Inference Time (ms)     | The total time taken to complete a single inference, including data preprocessing and postprocessing. |
| Throughput (inferences/sec) | The number of inferences the model can perform per second. |
| Memory Footprint (MB)   | The total amount of memory used by the model during execution. |
| Error Rate (%)          | The percentage of incorrect predictions made by the model. |
| Deployment Time (s)     | The time taken to deploy the model onto the device. |

| **Target HW/Platform Watch Metrics** | **Description**     |
|--------------------------------------|---------------------|
|            | Monitors the hardware platform's performance and resource usage. |
| RAM (KB)                             | The amount of random-access memory required by the model, measured in kilobytes. |
| Flash (KB)                           | The amount of flash memory required by the model, measured in kilobytes. |
| Computing Workload (%): CPU, GPU, NPU, TPU, VPU | The percentage of computing resources utilized by the model on different processing units for image classification. |
| Computing Speed/Clock (MHz)          | The operating frequency of the processing units for image classification, measured in megahertz. |
| Energy Consumption (mWh)             | Measures the amount of energy consumed by the model during inference. |
| Temperature (°C)                     | The operating temperature of the device while running the model. |
| Startup Time (ms)                    | The time taken for the model to load and be ready for inference. |
| Model Complexity (FLOPs)             | The number of floating-point operations required for a single inference. |


## Model Benchmarking

### **Benchmark Table Format** 

Run the model on any edge device/sim tool, to highlight its `performance` and `usability`.  

| **Model**          | **Task**             | **Accuracy**         | **Latency (ms)** | **Model Size (MB)** | **Platform**         | **References**                   |  
|---------------------|----------------------|----------------------|------------------|---------------------|----------------------|-----------------------------------|  
| MobileNet V2        | Image Classification | 72.0%                | 25               | 4.3                 | Android, iOS, Web    | [TensorFlow Lite](https://www.tensorflow.org/lite) |  

### Benchmarking Tools

This a list of go-to tools for benchmarking small AI models seamlessly. These tools cater to a variety of tasks and platforms, from edge devices to desktop environments:


### **1. General-Purpose Model Benchmarking Tools**
#### **[ONNX Runtime](https://onnxruntime.ai/)**
- **Use Case**: Benchmark ONNX models on multiple platforms (CPU, GPU, ARM).  
- **Features**:  
  - Highly optimized inference for small models.  
  - Built-in performance profiling tools.  
- **Integration**: Python, C++, JavaScript.  
- **Best For**: Cross-platform benchmarking.

#### **[TensorFlow Lite Benchmark Tool](https://www.tensorflow.org/lite/performance/measurement)**  
- **Use Case**: Benchmark TFLite models on mobile and edge devices.  
- **Features**:  
  - Latency, memory usage, and CPU/GPU acceleration profiling.  
  - Support for Android and iOS.  
- **Integration**: CLI-based tool; Android APK available for mobile testing.  
- **Best For**: Android/iOS mobile inference.

#### **[PyTorch Benchmark Utils](https://pytorch.org/tutorials/recipes/benchmark.html)**  
- **Use Case**: Benchmark PyTorch models for speed and memory.  
- **Features**:  
  - Granular performance tracking using `torch.utils.benchmark`.  
  - CUDA optimization testing.  
- **Integration**: Python API.  
- **Best For**: Development-phase PyTorch models.

---

### **2. Edge and Embedded Device-Specific Tools**  
#### **[Edge Impulse](https://www.edgeimpulse.com/)**  
- **Use Case**: End-to-end benchmarking for embedded ML.  
- **Features**:  
  - Automated profiling for ARM Cortex-M, NVIDIA Jetson, and more.  
  - Power consumption analysis.  
- **Integration**: Cloud platform and SDK for local benchmarking.  
- **Best For**: IoT and embedded use cases.

#### **[AIPerf (AI Performance Tool)](https://github.com/AIPerf/ai-performance-tool)**  
- **Use Case**: Evaluate edge AI model performance on specific devices.  
- **Features**:  
  - Latency and throughput measurement.  
  - Supports TensorFlow Lite, ONNX, and PyTorch.  
- **Integration**: CLI tool.  
- **Best For**: Quick comparisons across hardware.

#### **[DeepSparse](https://github.com/neuralmagic/deepsparse)**  
- **Use Case**: Benchmark sparsified models for CPU inference.  
- **Features**:  
  - Extreme CPU optimization for quantized models.  
  - Integration with ONNX models.  
- **Integration**: Python SDK.  
- **Best For**: Desktop or server-based CPU performance.

---

### **3. Vision-Specific Tools**  
#### **[OpenVINO Benchmark Tool](https://docs.openvino.ai/2023.0/benchmark_tool.html)**  
- **Use Case**: Benchmark vision models optimized for Intel hardware.  
- **Features**:  
  - Detailed profiling of latency, throughput, and layer performance.  
  - Runs on CPUs, GPUs, and VPUs.  
- **Integration**: CLI and Python API.  
- **Best For**: Intel-based edge devices.

#### **[TensorRT](https://developer.nvidia.com/tensorrt)**  
- **Use Case**: Benchmark NVIDIA-optimized models.  
- **Features**:  
  - High-speed inference profiling for deep learning models.  
  - GPU utilization and FP16/INT8 precision testing.  
- **Integration**: Python and C++ APIs.  
- **Best For**: NVIDIA Jetson and desktop GPUs.

#### **[Roboflow Benchmarking Tools](https://roboflow.com/)**  
- **Use Case**: Benchmark computer vision models for object detection and classification.  
- **Features**:  
  - Hosted dataset evaluation and accuracy benchmarking.  
  - Dataset preparation included.  
- **Integration**: Web-based and API support.  
- **Best For**: Model accuracy evaluation.

---

### **4. Multimodal Model Tools**  
#### **[Hugging Face Evaluate](https://huggingface.co/docs/evaluate/index)**  
- **Use Case**: Benchmark multimodal tasks (vision-language, text-audio).  
- **Features**:  
  - Built-in support for common metrics like BLEU, F1, and accuracy.  
  - Easy integration with Hugging Face models and datasets.  
- **Integration**: Python SDK.  
- **Best For**: Vision-language or text-image tasks.

#### **[MLPerf Tiny](https://mlcommons.org/en/inference-tiny/)**  
- **Use Case**: Benchmarks for embedded and small ML models.  
- **Features**:  
  - Open benchmarking suite with predefined scenarios.  
  - Cross-platform and model-agnostic.  
- **Integration**: CLI-based.  
- **Best For**: Standardized comparisons.

---

### **5. Quantization and Compression-Specific Tools**  
#### **[Neural Network Distiller](https://github.com/NervanaSystems/distiller)**  
- **Use Case**: Benchmark and analyze compressed models.  
- **Features**:  
  - Sparsity, pruning, and quantization support.  
  - Layer-wise performance analysis.  
- **Integration**: Python API.  
- **Best For**: Model compression benchmarking.

#### **[NNI (Neural Network Intelligence)](https://github.com/microsoft/nni)**  
- **Use Case**: Benchmark models with automated quantization and tuning.  
- **Features**:  
  - Automated performance optimization.  
  - Cross-framework compatibility.  
- **Integration**: Python SDK.  
- **Best For**: Experimentation with small models.

---

## Resources

### **Tips for Seamless Benchmarking**  
- **Choose the Right Hardware**: Match the device to the model's deployment environment.  
- **Automate with Scripts**: Use tools with Python/CLI APIs to run repeatable tests.  
- **Measure Across Scenarios**: Include latency, throughput, memory usage, and accuracy metrics.  
- **Document Results**: Store benchmarks in a standardized format (e.g., Markdown tables).  
