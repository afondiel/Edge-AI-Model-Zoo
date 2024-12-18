# **Model Evaluation & Benchmarking Guide**

## **Overview**

This guideline provides a comprehensive framework for selecting a suitable machine learning model tailored to Edge devices based on hardware platforms and real-world applications.


## Table of Contents

- [Overview](#overview)
- [Evaluation Criteria & Key Metrics](#evaluation-criteria--key-metrics)
  - [1. Computational Efficiency](#1-computational-efficiency)
  - [2. Model Size](#2-model-size)
  - [3. Memory Footprint](#3-memory-footprint)
  - [4. Resolution vs. Performance Trade-off](#4-resolution-vs-performance-trade-off)
  - [5. Latency and Real-Time Capability](#5-latency-and-real-time-capability)
  - [6. Energy Efficiency](#6-energy-efficiency)
  - [7. Deployment Constraints](#7-deployment-constraints)
  - [8. Accuracy vs. Resource Trade-Off](#8-accuracy-vs-resource-trade-off)
- [Benchmarking: Target HW Platform Profiling](#benchmarking-target-hw-platform-profiling)
- [Benchmarking: Target Model Profiling](#benchmarking-target-model-profiling)
- [Best Practices & Considerations](#best-practices--considerations)
- [Practical Steps for Optimization](#practical-steps-for-optimization)



## **Evaluation Criteria & Key Metrics**

When choosing a machine learning model for edge devices (e.g., Mobiles, IoT devices, embedded systems), consider these key factors:

### **1. Computational Efficiency**
- **Key Metric**: **Operations Per Second (OPS)**  
  - Opt for models with low OPS to minimize latency and power consumption, particularly on CPUs.
  - This is critical for real-time applications like object detection or pose estimation.

### **2. Model Size**
- **Key Metric**: **Model Size (MB)**  
  - Smaller models reduce memory and storage demands, enabling deployment on resource-constrained devices.
  - Compact models are ideal for scenarios requiring frequent updates or over-the-air (OTA) deployments.

### **3. Memory Footprint**
- **Key Metric**: **Parameters (M)**  
  - Models with fewer parameters consume less memory during inference, essential for devices with limited RAM.
  - Excessive parameters can lead to slower CPU performance.

### **4. Resolution vs. Performance Trade-off**
- **Key Metric**: **Input Resolution**  
  - Higher resolutions improve accuracy but require more computational resources.
  - For edge devices, prefer low-resolution models unless high accuracy is indispensable.

### **5. Latency and Real-Time Capability**
- Low-latency predictions are crucial for real-time applications such as autonomous vehicles, surveillance, and robotics.
- Benchmark models under expected workloads to ensure acceptable response times.

### **6. Energy Efficiency**
- For battery-powered or energy-constrained devices, prioritize models with low OPS and reduced computational overhead.

### **7. Deployment Constraints**
- **Device Constraints**: Consider the CPU architecture, memory, storage, and thermal limitations of the device.
- **Framework Compatibility**: Ensure compatibility with frameworks like TensorFlow Lite, ONNX, or OpenVINO.

### **8. Accuracy vs. Resource Trade-Off**
- Select models that balance acceptable accuracy with edge resource limitations.
- Benchmark multiple models to meet the target application's accuracy requirements.


## **Benchmarking: Target HW Platform Profiling**

Selecting the right edge device is crucial for the overall performance and efficiency of your application. Below is a performance template to help guide you in choosing the appropriate edge platform based on your application requirements (e.g., computational demands, environmental conditions, operational constraints, etc.). Here’s a practical [example](https://github.com/afondiel/Edge-AI-Platforms).


| **Target Device**       | **Model Specs**     | **Accuracy (%)** | **Latency (ms)** | **Throughput (inf/s)** | **Memory Usage (RAM)** | **Energy (mWh)** | **Startup Time (ms)** | **Temperature (°C)** | **Complexity (OPS)** | **Compute Speed (MHz/GHz)** | **Deployment Time (s)** |
|--------------------------------|---------------------|------------------|------------------|------------------------|------------------------|------------------|------------------------|----------------------|--------------------------|-----------------------------|--------------------------|
| **System on Chips (SoCs)**     | Size (MB)           | %                | ms               | Inferences/sec         | MB                    | mWh              | ms                     | °C                   | GFlops                  | MHz/GHz                    | s                        |
| **Microcontrollers (MCUs)**    | Size (KB)           | %                | ms               | Inferences/sec         | KB                    | mWh              | ms                     | °C                   | GFlops                  | MHz                       | s                        |
| **Field-Programmable Gate Arrays (FPGAs)** | Size (MB) | %                | µs/ms            | Inferences/sec         | MB                    | mWh              | ms                     | °C                   | GFlops                  | MHz                       | s                        |
| **Edge AI Boxes and Gateways** | Size (MB/GB)        | %                | ms               | Inferences/sec         | MB/GB                 | mWh/W            | ms                     | °C                   | GFlops                  | MHz/GHz                    | s                        |
| **Mobile and Embedded Devices** | Size (MB/GB)       | %                | ms               | Inferences/sec         | MB/GB                 | mWh/W            | ms                     | °C                   | GFlops                  | GHz                       | s                        |
| **Specialized Edge Devices**   | Size (MB/GB)        | %                | ms               | Inferences/sec         | MB/GB                 | mWh/W            | ms                     | °C                   | GFlops                  | GHz                       | s                        |
| **Industrial and Custom Edge Devices** | Size (MB/GB) | %                | ms               | Inferences/sec         | MB/GB                 | mWh/W            | ms                     | °C                   | GFlops                  | GHz                       | s                        |
| **Robotics-Focused Edge Devices** | Size (MB/GB)     | %                | ms               | Inferences/sec         | MB/GB                 | mWh/W            | ms                     | °C                   | GFlops                  | GHz                       | s                        |

### **Rule of Thumb**
- **Small Devices (e.g., IoT sensors)**: Use the smallest model with acceptable accuracy.
- **Mid-Power Devices (e.g., drones, edge servers)**: Prefer low-resolution models with moderate complexity.
- **High-Critical Applications (e.g., autonomous vehicles)**: Prioritize models balancing accuracy and latency.


## **Benchmarking: Target Model Profiling**

Here, the goal is to analyze the model performance to identify bottlenecks and optimize its execution time and memory usage to meet the real-time application requirements.

### **How It Works**

1. **Download and Evaluate**: Test the model on a standardized dataset (e.g., ImageNet, COCO).
2. **Measure Performance**:  
   - Use tools like ONNX Runtime, TensorFlow Lite Benchmark Tool, or CoreML Profiler.  
   - Test latency on edge devices (e.g., Raspberry Pi, Jetson Nano).
3. **Document Results**: Log metrics in a benchmark table.
4. **Submit with PR**: Attach benchmark results to your pull request.

### **Benchmark Table Example**

**Short version**:

| **Model**       | **Task**               | **Accuracy** | **Latency (ms)** | **Model Size (MB)** | **Platform**         | **Reference**                         |
|------------------|------------------------|--------------|------------------|---------------------|----------------------|---------------------------------------|
| MobileNet V2     | Image Classification  | 72.0%        | 25               | 4.3                 | Android, iOS, Web    | [TensorFlow Lite](https://www.tensorflow.org/lite) |

**Long version:**

| **Model**   | **Size (MB)** | **Top-1 Accuracy** | **Top-5 Accuracy** | **Parameters** | **Depth** | **Time (ms) per inference step (CPU)** | **Time (ms) per inference step (GPU)** |
|-------------|---------------|---------------------|---------------------|----------------|-----------|-----------------------------------------|-----------------------------------------|
| Xception    | 88            | 79.0%              | 94.5%              | 22.9M          | 81        | 109.4                                   | 8.1                                     |
| VGG16       | 528           | 71.3%              | 90.1%              | 138.4M         | 16        | 69.5                                    | 4.2                                     |
| VGG19       | 549           | 71.3%              | 90.0%              | 143.7M         | 19        | 84.8                                    | 4.4                                     |
| ResNet50    | 98            | 74.9%              | 92.1%              | 25.6M          | 107       | 58.2                                    | 4.6                                     |

**Test environments**
```
The top-1 and top-5 accuracy refers to the model's performance on the ImageNet validation dataset.

Depth refers to the topological depth of the network. This includes activation layers, batch normalization layers etc.

Time per inference step is the average of 30 batches and 10 repetitions.

- CPU: AMD EPYC Processor (with IBPB) (92 core)
- RAM: 1.7T
- GPU: Tesla A100
- Batch size: 32
Depth counts the number of layers with parameters.
```
(Source: [Keras Applications](https://keras.io/api/applications/))


### **Benchmarking Tools**

## **Benchmarking Tools**

<details>
  <summary>1. General-Purpose Tools</summary>

  ##### **[ONNX Runtime](https://onnxruntime.ai/)**
  - **Use Case**: Benchmark ONNX models across platforms (CPU, GPU, ARM).
  - **Features**:  
    - Optimized inference for small models.  
    - Built-in performance profiling.
  - **Best For**: Cross-platform benchmarking.

  ##### **[TensorFlow Lite Benchmark Tool](https://www.tensorflow.org/lite/performance/measurement)**
  - **Use Case**: Profile TFLite models on mobile/edge devices.
  - **Features**:  
    - Profiling latency, memory, and acceleration.
    - Android and iOS support.
  - **Best For**: Mobile inference.

  ##### **[PyTorch Benchmark Utilities](https://pytorch.org/tutorials/recipes/benchmark.html)**
  - **Use Case**: Benchmark PyTorch models for speed and memory usage.
  - **Features**:  
    - Granular performance tracking with `torch.utils.benchmark`.  
    - CUDA optimization testing.
  - **Best For**: Development-phase PyTorch models.

</details>

<details>
  <summary>2. Edge Devices</summary>

  ##### **[Edge Impulse](https://www.edgeimpulse.com/)**
  - **Use Case**: Benchmark models specifically designed for edge devices, including MCUs and constrained hardware.  
  - **Features**:  
    - Real-time inference profiling on low-power devices.  
    - End-to-end platform for training, optimizing, and deploying models.  
    - Built-in support for sensor data.  
  - **Integration**: Web-based interface and CLI tools.  
  - **Best For**: End-to-end benchmarking and deployment on edge devices.

  ##### **[NVIDIA Jetson Performance Tool (JTOP)](https://github.com/rbonghi/jetson_stats)**
  - **Use Case**: Monitor and benchmark AI workloads on NVIDIA Jetson devices.  
  - **Features**:  
    - GPU and CPU resource usage tracking.  
    - Memory, temperature, and power profiling.  
  - **Integration**: Command-line tool for Jetson devices.  
  - **Best For**: Profiling heavy AI models on NVIDIA edge platforms.

  ##### **[ARM Compute Library](https://developer.arm.com/tools-and-software/compute-library)**
  - **Use Case**: Benchmark AI models on ARM processors (e.g., Cortex-M, Cortex-A).  
  - **Features**:  
    - Optimized for edge inference on ARM architecture.  
    - Real-time latency and memory profiling.  
  - **Integration**: C++ library with Python bindings.  
  - **Best For**: Models running on ARM-based IoT and mobile devices.

</details>

<details>
  <summary>3. Cloud-Based Benchmarking Tools</summary>

  ##### **[AWS SageMaker Edge Manager](https://aws.amazon.com/sagemaker/edge/)**
  - **Use Case**: Benchmark, manage, and optimize AI models on edge devices connected to AWS.  
  - **Features**:  
    - Monitor device performance remotely.  
    - Deploy models with integrated latency and resource tracking.  
  - **Integration**: AWS ecosystem.  
  - **Best For**: Large-scale edge device deployment and monitoring.

  ##### **[Google ML Kit](https://developers.google.com/ml-kit)**
  - **Use Case**: Benchmark pre-trained models and custom solutions on Android/iOS devices.  
  - **Features**:  
    - Lightweight on-device inference profiling.  
    - Extensive documentation and pre-built APIs for vision tasks.  
  - **Integration**: Android/iOS SDK.  
  - **Best For**: Real-time applications on mobile devices.

</details>


## **Best Practices & Considerations**

### Model Evaluation:
- **Target Metrics First**: Define critical metrics like latency, memory footprint, or accuracy based on the application.  
- **Test Across Devices**: Models may perform differently on diverse edge hardware—ensure compatibility.  
- **Use Representative Data**: Benchmark using datasets that match real-world scenarios to achieve accurate evaluations.  
- **Automate the Process**: Set up automated pipelines for benchmarking multiple models to save time.  
- **Iterate and Optimize**: Use optimization techniques (e.g., pruning, quantization) to balance accuracy and efficiency.

### Common Pitfalls to Avoid:
- **Overlooking Real-World Constraints**: Failing to account for intermittent power, connectivity, or thermal limitations can derail deployment.  
- **Underestimating Deployment Complexity**: Ensure the model integrates seamlessly with existing frameworks and workflows.  
- **Ignoring Edge-Specific Tools**: Generic tools might not capture nuances of edge environments. Leverage specialized tools for accurate profiling.  

### Some Real-Life Use Cases:
- **Autonomous Drones**: EfficientNet-lite quantized to INT8 for low-latency object detection.  
- **IoT Sensors**: TinyML model for temperature anomaly detection on Cortex-M MCUs.  
- **Retail Analytics**: MobileNet V3 for real-time customer tracking on Android edge devices.  

## **Practical Steps for Optimization**
1. **Benchmark Models**: Evaluate latency, memory usage, and energy efficiency on target edge hardware.
2. **Choose Lightweight Architectures**: Use models like MobileNet, EfficientNet-Lite, or low-resolution versions of larger models.
3. **Quantize Models**: Convert models to INT8 or lower precision to reduce size and computation.
4. **Simulate Edge Scenarios**: Test models under realistic conditions (e.g., low power, intermittent connectivity).
5. **Optimize Pipelines**: Employ pruning, knowledge distillation, or layer fusion to enhance performance.