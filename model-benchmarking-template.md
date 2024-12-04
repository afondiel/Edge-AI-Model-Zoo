# Models Benchmarking Template and Profiling Metrics

## **How it works?**

1. **Download and Evaluate**: Run the model on a standardized dataset (e.g., ImageNet, COCO).  
2. **Measure Performance**:  
   - Use tools like ONNX Runtime, TensorFlow Lite Benchmark Tool, or CoreML `Profiler`.  
   - Record latency on mobile/edge devices (e.g., Raspberry Pi, Jetson Nano).  
3. **Document Results**: Use the table above to document key metrics.  
4. **Submit with PR**: Attach the benchmark results with your PR submission. 

## **Benchmark Table Format** 

Run the model on any edge device/sim tool, to highlight its `performance` and `usability`.  

| **Model**          | **Task**             | **Accuracy**         | **Latency (ms)** | **Model Size (MB)** | **Platform**         | **References**                   |  
|---------------------|----------------------|----------------------|------------------|---------------------|----------------------|-----------------------------------|  
| MobileNet V2        | Image Classification | 72.0%                | 25               | 4.3                 | Android, iOS, Web    | [TensorFlow Lite](https://www.tensorflow.org/lite) |  

## **Profiling Key Metrics (General Template)**

The goal is to optimize the model performance to meet the HW Platform requirements (Hardware-aware).

**### Edge Devices**

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

**### MCUs**

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
