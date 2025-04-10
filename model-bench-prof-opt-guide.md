# Model Benchmarking, Profiling and Optimization Guide

This guide walks you through benchmarking, profiling, and optimizing edge AI models to ensure they meet your application’s performance needs. It’s designed for users who’ve selected a model from the [Edge-AI-Model-Zoo/README.md](./README.md) and want to evaluate and enhance it.

## Table of Contents
- [Workflow](#workflow)
- [Step 1: Benchmark Your Model](#step-1-benchmark-your-model)
- [Step 2: Profile and Identify Bottlenecks](#step-2-profile-and-identify-bottlenecks)
- [Step 3: Optimize and Compare](#step-3-optimize-and-compare)
- [Tools and Frameworks](#tools-and-frameworks)
- [Resources](#resources)

## Workflow

```mermaid
   graph LR
       A[Choose Model] --> B[Benchmark]
       B --> C[Profile]
       C --> D[Optimize]
       D --> E[Compare]
```

## Step 1: Benchmark Your Model
Evaluate your model’s baseline performance on your target edge device.

### Key Metrics
- **Latency (ms)**: Time per inference.
- **Throughput (inf/s)**: Inferences per second.
- **Memory Usage (MB)**: RAM and storage footprint.
- **Energy (mWh)**: Power consumption.

### Example
```python
# Benchmark latency with TensorFlow Lite
import tensorflow as tf
import time

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

start_time = time.time()
interpreter.invoke()
latency = (time.time() - start_time) * 1000  # ms
print(f"Latency: {latency:.2f} ms")
```
- Predict and benchmark `throughput` using [tensorflow](https://www.tensorflow.org/):
```python
# Predict and benchmark throughput using tensorflow
def predict_and_benchmark_throughput(batched_input, infer, N_warmup_run=50, N_run=1000):

  elapsed_time = []
  all_preds = []
  batch_size = batched_input.shape[0]

  for i in range(N_warmup_run):
    labeling = infer(batched_input)
    preds = labeling['output_0'].numpy()

  for i in range(N_run):
    start_time = time.time()
    labeling = infer(batched_input)
    preds = labeling['output_0'].numpy()
    end_time = time.time()
    elapsed_time = np.append(elapsed_time, end_time - start_time)
    all_preds.append(preds)

    if i % 50 == 0:
      print('Steps {}-{} average: {:4.1f}ms'.format(i, i+50, (elapsed_time[-50:].mean()) * 1000))

  print('Throughput: {:.0f} images/s'.format(N_run * batch_size / elapsed_time.sum()))
  return all_preds
```
- `Accuracy`:

```python

# Observe accuracy using tensorflow
def show_predictions(model):

  img_path = './data/img0.JPG'  # golden_retriever
  img = image.load_img(img_path, target_size=(299, 299))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  x = tf.constant(x)

  labeling = model(x)
  preds = labeling['predictions'].numpy()

  # decode the results into a list of tuples (class, description, probability)
  # (one such list for each sample in the batch)
  print('{} - Predicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))
  plt.subplot(2,2,1)
  plt.imshow(img);
  plt.axis('off');
  plt.title(decode_predictions(preds, top=3)[0][0][1])
```

### Choose Your Target Edge Platform carefully

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

## Step 2: Profile and Identify Bottlenecks

Analyze where your model spends time and resources.

Here, the goal is to analyze the model performance to identify bottlenecks and optimize its execution time and memory usage to meet the real-time application requirements.

### Tools

- **TensorFlow Lite Benchmark Tool**: Measures latency and memory on mobile devices.
- **ONNX Runtime Profiler**: Profiles layer-wise performance.
- **NVIDIA JTOP**: Monitors Jetson device resources.

### **Profiling Metrics Examples**

**Short version**:

| **Model**       | **Task**               | **Accuracy** | **Latency (ms)** | **Model Size (MB)** |**Params Size (MB)** | **Platform**         | **Reference**                         |
|------------------|------------------------|--------------|------------------|---------------------|---------------------|----------------------|---------------------------------------|
| MobileNet V2     | Image Classification  | 72.0%        | 25               | 4.3                 |3.5                 | Android, iOS, Web    | [TensorFlow Lite](https://www.tensorflow.org/lite) |

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


## Step 3: Optimize and Compare
Apply optimizations and compare performance before and after.

### Optimization Techniques
1. **Quantization**: Reduce precision (e.g., FP32 → INT8).
2. **Pruning**: Remove unnecessary weights.
3. **Layer Fusion**: Combine operations for efficiency.

### Example Comparison
| Metric | Before | After (Quantized) |
|--------|--------|-------------------|
| Latency (ms) | 25 | 18 |
| Model Size (MB) | 4.3 | 1.2 |
| Accuracy (%) | 72 | 70 |


## **Tools and Frameworks**

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

## **Pro Tips**

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

### **Practical Steps for Optimization**
1. **Benchmark Models**: Evaluate latency, memory usage, and energy efficiency on target edge hardware.
2. **Choose Lightweight Architectures**: Use models like MobileNet, EfficientNet-Lite, or low-resolution versions of larger models.
3. **Quantize Models**: Convert models to INT8 or lower precision to reduce size and computation.
4. **Simulate Edge Scenarios**: Test models under realistic conditions (e.g., low power, intermittent connectivity).
5. **Optimize Pipelines**: Employ pruning, knowledge distillation, or layer fusion to enhance performance.

### Iterate
- Test on real hardware, not just simulators.
- Use representative datasets for accurate benchmarking.
- Iterate: Optimize incrementally and re-benchmark.

## Resources
- **Benchmarking Tools**:
   - [TensorFlow Lite](https://www.tensorflow.org/lite/performance/measurement)
   - [ONNX Runtime](https://onnxruntime.ai/)
- **Hardware Profiling**:
   - [Qualcomm Profiler](https://www.qualcomm.com/developer/software/qualcomm-profiler)
   - [Edge Impulse](https://www.edgeimpulse.com/)
   - [Geekbench - a cross-platform benchmark that measures your system's performance](https://www.geekbench.com/)
- **Optimization Guides**: [Hugging Face Optimization](https://huggingface.co/docs/transformers/performance)
