[![](https://img.shields.io/badge/Contribute-Welcome-green)](./CONTRIBUTING.md)

# Edge-AI Model Zoos

A curated list of Model Zoos & Hubs where you can find production-ready and optimized models for resource-constrained devices.

## Table of Contents
1. [Model Zoos & Hubs](#1-model-zoos--hubs)
2. [Model by Domain & Use Case](#2-model-by-domain--use-case)
3. [Resources](#resources)
    - [How to choose the best model for Edge AI application](#how-to-choose-the-best-model-for-edge-ai-application)
    - [Edge AI Technical Guide for Developers and Practitioners](#edge-ai-technical-guide-for-developers-and-practitioners)

## 1. Model Zoos & Hubs

[Back to Table of Contents](#table-of-contents)

| Model Zoo | Description | Links |
|------------------------|---------------------------------------------------------------------|----------------------------------------------------|
| Edge AI Labs Model Zoo | A collection of pre-trained, optimized models for low-power devices.| [EdgeAI Labs](https://edgeai.modelnova.ai/models/) |
| Edge Impulse Model Zoo | A repository of models optimized for edge devices. | [Edge Impulse Model Zoo](https://www.edgeimpulse.com/) |
| ONNX Model Zoo | A collection of pre-trained, state-of-the-art models in the ONNX format. | [ONNX Model Zoo](https://github.com/onnx/models) |
| NVIDIA Pretrained AI Models (NGC + TAO)| Accelerate AI development with world-class customizable pretrained models from NVIDIA. | - [NVIDIA Pretrained AI Models - Main](https://developer.nvidia.com/ai-models) <br> - [NGC Model Catalog](https://catalog.ngc.nvidia.com/models?filters=&orderBy=weightPopularDESC&query=&page=&pageSize=) <br> -  [TAO Model Zoo](https://docs.nvidia.com/tao/tao-toolkit/text/model_zoo/overview.html)|
| OpenVINO Model Zoo | A collection of pre-trained models ready for use with Intel's OpenVINO toolkit. | [OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) |
| Qualcomm Model Zoo | A collection of AI models from Qualcomm | - [Qualcomm AI Hub Models](https://github.com/quic/ai-hub-models/) <br> - [AIMET Model Zoo](https://github.com/quic/aimet-model-zoo)|
| Core ML Models | A collection of AI models for Apple devices | [Core ML Models](https://developer.apple.com/machine-learning/models/) |
| LiteRT Pre-trained models | Pre-trained models optimized for Google's Lite Runtime. | [LiteRT Pre-trained Models](https://ai.google.dev/edge/litert/models/trained) |
| Keras Applications | Pre-trained models for Keras applications| [Keras Pre-trained Models](https://keras.io/api/applications/#available-models) |
| MediaPipe | Framework for building multimodal applied machine learning pipelines. | [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) |
| TensorFlow Model Garden | A repository with a collection of TensorFlow models. | [TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master) |
| Pytorch Model Zoo | A hub for pre-trained models on PyTorch framework. | [Pytorch Model Zoo](https://pytorch.org/serve/model_zoo.html) |
| stm32ai-modelzoo | AI Model Zoo for STM32 microcontroller devices. | [stm32ai-modelzoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) |
| Model Zoo | A collection of pre-trained models for various machine learning tasks. | [Model Zoo](https://modelzoo.co/) |
| Hugging Face Models| A collection of pre-trained models for various machine learning tasks. | [Hugging Face Models](https://huggingface.co/models) |
| Papers with Code | A repository that links academic papers to their respective code and models. | [Papers with Code](https://paperswithcode.com/) |
| MXNet Model Zoo | A collection of pre-trained models for the Apache MXNet framework. | [MXNet Model Zoo](https://mxnet.apache.org/versions/1.1.0/model_zoo/index.html) |
| Deci’s Model Zoo | A curated list of high-performance deep learning models. | [Deci’s Model Zoo](https://deci.ai/modelzoo/) |
| Jetson Model Zoo and Community Projects | NVIDIA's collection of models and projects for Jetson platform. | [Jetson Model Zoo and Community Projects](https://developer.nvidia.com/embedded/community/jetson-projects) |
| Magenta | Models for music and art generation from Google's Magenta project. | [Magenta](https://github.com/magenta/magenta/tree/main/magenta/models/arbitrary_image_stylization) |
| Awesome-CoreML-Models Public | A collection of CoreML models for iOS developers. | [Awesome-CoreML-Models Public](https://github.com/likedan/Awesome-CoreML-Models) |
| Pinto Models | A variety of models for computer vision tasks. | [Pinto Models](https://github.com/PINTO0309/PINTO_model_zoo) |
| Baidu AI Open Model Zoo | Baidu's collection of AI models. | [Baidu AI Open Model Zoo](https://ai.baidu.com/tech/modelzoo) |
| Hailo Model Zoo | A set of models optimized for Hailo's AI processors. | [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) |

## 2. Model by Domain & Use Case

[Back to Table of Contents](#table-of-contents)

This is a non-exhaustive selection of models from several platforms listed in [Section 1](#1-model-zoos--hubs), ranged into six domains and a variety of tasks, with a focus on efficiency and real-world applications.

<table border="1">
  <thead>
    <tr>
      <th>Domain</th>
      <th>Task</th>
      <th>Model</th>
      <th>Description</th>
      <th>Reference</th>
    </tr>
  </thead>
  <tbody>
    <!-- Computer Vision: 10 rows -->
    <tr>
      <td rowspan="10">Computer Vision</td>
      <td>Object Detection</td>
      <td>yolov8_det</td>
      <td>Object detection for edge devices</td>
      <td><a href="https://github.com/ultralytics/yolov8">YOLOv8 on GitHub</a></td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mobilenet_v3_small</td>
      <td>Lightweight image classification</td>
      <td><a href="https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5">MobileNetV3 on TensorFlow Hub</a></td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>deeplabv3_resnet50</td>
      <td>Semantic image segmentation</td>
      <td><a href="https://tfhub.dev/tensorflow/deeplabv3/1">DeepLabV3 on TensorFlow Hub</a></td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>yolov8_seg</td>
      <td>Object detection and segmentation</td>
      <td><a href="https://github.com/ultralytics/yolov8">YOLOv8 on GitHub</a></td>
    </tr>
    <tr>
      <td>Object Tracking</td>
      <td>DeepSort</td>
      <td>Real-time object tracking</td>
      <td><a href="https://github.com/nwojke/deep_sort">DeepSort on GitHub</a></td>
    </tr>
    <tr>
      <td>Pose Estimation</td>
      <td>openpose</td>
      <td>Human pose estimation</td>
      <td><a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose">OpenPose on GitHub</a></td>
    </tr>
    <tr>
      <td>Facial Recognition</td>
      <td>mediapipe_face</td>
      <td>Face detection and recognition</td>
      <td><a href="https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html">MediaPipe Face on Google AI</a></td>
    </tr>
    <tr>
      <td>Optical Character Recognition</td>
      <td>trocr</td>
      <td>Text recognition in images</td>
      <td><a href="https://huggingface.co/microsoft/trocr-base">TrOCR on Hugging Face</a></td>
    </tr>
    <tr>
      <td>Video Classification</td>
      <td>resnet_2plus1d</td>
      <td>Video classification for action recognition</td>
      <td><a href="https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/">ResNet-2+1D on PyTorch Hub</a></td>
    </tr>
    <tr>
      <td>Video Classification</td>
      <td>resnet_3d</td>
      <td>3D CNN for video classification</td>
      <td><a href="https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/">ResNet-3D on PyTorch Hub</a></td>
    </tr>
    <!-- Audio Processing: 10 rows -->
    <tr>
      <td rowspan="10">Audio Processing</td>
      <td>Speech-to-Text</td>
      <td>distil-whisper</td>
      <td>Lightweight speech recognition model</td>
      <td><a href="https://huggingface.co/distil-whisper/distil-large-v3">Distil-Whisper on Hugging Face</a></td>
    </tr>
    <tr>
      <td>Sound Classification</td>
      <td>audio-spectrogram-transformer</td>
      <td>Transformer for audio classification</td>
      <td><a href="https://huggingface.co/MIT/ast-finetuned-audioset">AST on Hugging Face</a></td>
    </tr>
    <tr>
      <td>Voice Activity Detection</td>
      <td>silero-vad</td>
      <td>Voice activity detection for edge devices</td>
      <td><a href="https://github.com/snakers4/silero-vad">Silero VAD on GitHub</a></td>
    </tr>
    <tr>
      <td>Acoustic Scene Classification</td>
      <td>panns</td>
      <td>Audio tagging and scene classification</td>
      <td><a href="https://github.com/qiuqiangkong/panns">PANNS on GitHub</a></td>
    </tr>
    <tr>
      <td>Speaker Diarization</td>
      <td>pyannote-audio</td>
      <td>Speaker diarization and segmentation</td>
      <td><a href="https://huggingface.co/pyannote/speaker-diarization">PyAnnote on Hugging Face</a></td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>wav2vec2</td>
      <td>Self-supervised speech representation learning</td>
      <td><a href="https://huggingface.co/facebook/wav2vec2-base-960h">Wav2vec2 on Hugging Face</a></td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <!-- Time Series: 10 rows -->
    <tr>
      <td rowspan="10">Time Series</td>
      <td>Predictive Maintenance</td>
      <td>tsmixer</td>
      <td>Time-series forecasting for maintenance</td>
      <td><a href="https://github.com/google-research/timesfm">TimesFM on GitHub</a></td>
    </tr>
    <tr>
      <td>Anomaly Detection</td>
      <td>IsolationForest</td>
      <td>Anomaly detection in time-series data</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html">IsolationForest on Scikit-learn</a></td>
    </tr>
    <tr>
      <td>Forecasting</td>
      <td>informer</td>
      <td>Transformer-based time-series forecasting</td>
      <td><a href="https://github.com/zhouhaoyi/Informer2020">Informer on GitHub</a></td>
    </tr>
    <tr>
      <td>Time-Series Classification</td>
      <td>rocket</td>
      <td>Efficient time-series classification</td>
      <td><a href="https://github.com/angus924/rocket">ROCKET on GitHub</a></td>
    </tr>
    <tr>
      <td>Image Super-Resolution</td>
      <td>real_esrgan_x4plus</td>
      <td>Image super-resolution for temporal data</td>
      <td><a href="https://github.com/xinntao/Real-ESRGAN">Real-ESRGAN on GitHub</a></td>
    </tr>
    <tr>
      <td>Image Inpainting</td>
      <td>lama_dilated</td>
      <td>Image inpainting for time-series analysis</td>
      <td><a href="https://github.com/saic-mdal/lama">LaMa on GitHub</a></td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <!-- NLP: 10 rows -->
    <tr>
      <td rowspan="10">NLP</td>
      <td>Speech Recognition</td>
      <td>Whisper</td>
      <td>General-purpose speech recognition model</td>
      <td><a href="https://huggingface.co/openai/whisper">Whisper on Hugging Face</a></td>
    </tr>
    <tr>
      <td>Keyword Spotting</td>
      <td>silero-kws</td>
      <td>Wake word detection for edge devices</td>
      <td><a href="https://github.com/snakers4/silero-models">Silero Models on GitHub</a></td>
    </tr>
    <tr>
      <td>Text Classification</td>
      <td>distilbert</td>
      <td>Lightweight transformer for text classification</td>
      <td><a href="https://huggingface.co/distilbert-base-uncased">DistilBERT on Hugging Face</a></td>
    </tr>
    <tr>
      <td>Named Entity Recognition</td>
      <td>bert-ner</td>
      <td>NER for entity extraction</td>
      <td><a href="https://huggingface.co/dslim/bert-base-NER">BERT-NER on Hugging Face</a></td>
    </tr>
    <tr>
      <td>Question Answering</td>
      <td>mobilebert</td>
      <td>Lightweight QA model for edge</td>
      <td><a href="https://huggingface.co/google/mobilebert-uncased">MobileBERT on Hugging Face</a></td>
    </tr>
    <tr>
      <td>Text Summarization</td>
      <td>bart</td>
      <td>Text summarization for short texts</td>
      <td><a href="https://huggingface.co/facebook/bart-base">BART on Hugging Face</a></td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <!-- Generative AI: Image Generation & Synthesis (7 rows) -->
    <tr>
      <td rowspan="21">Generative AI</td>
      <td rowspan="7">Image Generation & Synthesis</td>
      <td>ControlNet</td>
      <td>Fine control over image generation</td>
      <td><a href="https://github.com/lllyasviel/ControlNet">ControlNet on GitHub</a></td>
    </tr>
    <tr>
      <td>Stable Diffusion</td>
      <td>Text-to-image generation</td>
      <td><a href="https://huggingface.co/CompVis/stable-diffusion-v-1-4">Stable Diffusion on Hugging Face</a></td>
    </tr>
    <tr>
      <td>stylegan2</td>
      <td>Image generation</td>
      <td><a href="https://github.com/NVlabs/stylegan2">StyleGAN2 on GitHub</a></td>
    </tr>
    <tr>
      <td>Flux.1-schnell</td>
      <td>Fast text-to-image generation</td>
      <td><a href="https://huggingface.co/black-forest-labs/FLUX.1-schnell">Flux.1 on Hugging Face</a>, <a href="https://github.com/afondiel/awesome-smol">Awesome-Smol</a></td>
    </tr>
    <tr>
      <td>Reve</td>
      <td>Image generation with advanced text rendering</td>
      <td><a href="https://huggingface.co/reve/reve">Reve on Hugging Face</a></td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <!-- Generative AI: Small LLM (7 rows) -->
    <tr>
      <td rowspan="7">Small Language Model (SLM)</td>
      <td>SmolLM2-1.7B</td>
      <td>Small language model for efficient text generation</td>
      <td><a href="https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B">SmolLM2 on Hugging Face</a>, <a href="https://github.com/afondiel/awesome-smol">Awesome-Smol</a></td>
    </tr>
    <tr>
      <td>Gemma 2</td>
      <td>Lightweight open model for text generation</td>
      <td><a href="https://huggingface.co/google/gemma-2-2b">Gemma 2 on Hugging Face</a>, <a href="https://github.com/afondiel/awesome-smol">Awesome-Smol</a></td>
    </tr>
    <tr>
      <td>Phi-3.5-mini</td>
      <td>Small language model with strong reasoning</td>
      <td><a href="https://huggingface.co/microsoft/phi-3.5-mini-instruct">Phi-3.5-mini on Hugging Face</a>, <a href="https://github.com/afondiel/awesome-smol">Awesome-Smol</a></td>
    </tr>
    <tr>
      <td>Qwen2.5-1.5B</td>
      <td>Efficient language model for instruction following</td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-1.5B">Qwen2.5 on Hugging Face</a>, <a href="https://github.com/afondiel/awesome-smol">Awesome-Smol</a></td>
    </tr>
    <tr>
      <td>Mixtral-8x22B</td>
      <td>Sparse mixture of experts for text generation</td>
      <td><a href="https://huggingface.co/mixtral-8x22b">Mixtral on Hugging Face</a></td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <!-- Generative AI: Multimodality (7 rows) -->
    <tr>
      <td rowspan="7">Multimodality</td>
      <td>SmolVLM-256M</td>
      <td>Smallest vision-language model for image understanding</td>
      <td><a href="https://huggingface.co/HuggingFaceTB/SmolVLM-256M">SmolVLM-256M on Hugging Face</a>, <a href="https://github.com/afondiel/awesome-smol">Awesome-Smol</a></td>
    </tr>
    <tr>
      <td>SmolVLM-500M</td>
      <td>Vision-language model for image and text tasks</td>
      <td><a href="https://huggingface.co/HuggingFaceTB/SmolVLM-500M">SmolVLM-500M on Hugging Face</a>, <a href="https://github.com/afondiel/awesome-smol">Awesome-Smol</a></td>
    </tr>
    <tr>
      <td>BakLLaVA-1</td>
      <td>Multimodal model for text and image tasks</td>
      <td><a href="https://huggingface.co/SkunkworksAI/BakLLaVA-1">BakLLaVA-1 on Hugging Face</a>, <a href="https://github.com/afondiel/awesome-smol">Awesome-Smol</a></td>
    </tr>
    <tr>
      <td>PaliGemma</td>
      <td>Vision-language model for multimodal tasks</td>
      <td><a href="https://huggingface.co/google/paligemma">PaliGemma on Hugging Face</a></td>
    </tr>
    <tr>
      <td>Seed1.5-VL</td>
      <td>Vision-language model with strong multimodal performance</td>
      <td><a href="https://huggingface.co/ByteDance/Seed1.5-VL">Seed1.5-VL on Hugging Face</a></td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <!-- Misc: 10 rows -->
    <tr>
      <td rowspan="10">Misc</td>
      <td>Sensor Fusion</td>
      <td>mediapipe_pose</td>
      <td>Human pose estimation using sensor data</td>
      <td><a href="https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html">MediaPipe Pose on Google AI</a></td>
    </tr>
    <tr>
      <td>Activity Recognition</td>
      <td>har-cnn</td>
      <td>Human activity recognition from sensor data</td>
      <td><a href="https://github.com/saif-mahmud/human-activity-recognition">HAR-CNN on GitHub</a></td>
    </tr>
    <tr>
      <td>Contextual Awareness</td>
      <td>SmolVLM-256M</td>
      <td>Multimodal model for environment understanding</td>
      <td><a href="https://huggingface.co/HuggingFaceTB/SmolVLM-256M">SmolVLM-256M on Hugging Face</a>, <a href="https://github.com/afondiel/awesome-smol">Awesome-Smol</a></td>
    </tr>
    <tr>
      <td>Network Anomaly Detection</td>
      <td>LOF</td>
      <td>Local outlier factor for network anomalies</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html">LOF on Scikit-learn</a></td>
    </tr>
    <tr>
      <td>Device Behavior Anomaly</td>
      <td>Autoencoder</td>
      <td>Anomaly detection for device behavior</td>
      <td><a href="https://github.com/keras-team/keras-io/blob/master/examples/timeseries/timeseries_anomaly_detection.py">Keras Autoencoder</a></td>
    </tr>
    <tr>
      <td>Sensor Data Anomaly</td>
      <td>OC-SVM</td>
      <td>One-class SVM for sensor data anomalies</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html">OneClassSVM on Scikit-learn</a></td>
    </tr>
    <tr>
      <td>On-device Control Systems</td>
      <td>TD3</td>
      <td>Twin Delayed DDPG for control systems</td>
      <td><a href="https://github.com/sfujim/TD3">TD3 on GitHub</a></td>
    </tr>
    <tr>
      <td>Various</td>
      <td>pinecone</td>
      <td>Vector database</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Various</td>
      <td>weaviate-c2</td>
      <td>Vector database</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Various</td>
      <td>upstage</td>
      <td>Various models</td>
      <td><a href="https://github.com/afondiel/awesome-smol">Awesome-Smol</a></td>
    </tr>
  </tbody>
</table>

## Resources

### How to choose the best model for an Edge AI application

Selecting the right model for edge deployment is critical for balancing **performance**, **accuracy** and **efficiency**.

### Why It Matters
- **Efficiency**: Edge devices (e.g., IoT, mobile, embedded systems) have limited compute, memory, and power.
- **Performance**: Real-time applications (e.g., autonomous drones, smart cameras) demand low latency and high accuracy.
- **Scalability**: The right model ensures cost-effective deployment across devices.

### Key Criteria
1. **Task Requirements**: Match the model to your application (e.g., vision, audio, multimodal).
2. **Hardware Constraints**: Consider compute (OPS), memory (MB), and energy (mWh) limits of your device.
3. **Performance Goals**: Balance accuracy, latency, and throughput for your use case.
4. **Deployment Ease**: Check compatibility with frameworks (e.g., TensorFlow Lite, ONNX).

**Next Steps**: Once you’ve shortlisted a model, use the [Edge AI Benchmarking Guide](https://github.com/afondiel/Edge-AI-Benchmarking) to **profile** and **optimize** the model performance.

### Edge AI Technical Guide for Developers and Practitioners

- [Edge AI Engineering](https://github.com/afondiel/edge-ai-engineering) 
- [Edge AI Technical Guide](https://github.com/afondiel/computer-science-notebook/tree/master/core/systems/edge-computing/edge-ai/concepts)
- [Edge AI End-to-End Stack](https://www.qualcomm.com/developer/artificial-intelligence)
- [Edge AI Deployment Stack](https://github.com/afondiel/computer-science-notebook/tree/master/core/systems/edge-computing/edge-ai/concepts/deployment)
- [Edge AI Optimization Stack](https://github.com/afondiel/computer-science-notebook/tree/master/core/systems/edge-computing/edge-ai/concepts/optimization)
- [Edge AI Frameworks](https://github.com/afondiel/computer-science-notebook/tree/master/core/systems/edge-computing/edge-ai/frameworks)
- [Edge AI Model Zoos](https://github.com/afondiel/Edge-AI-Model-Zoo)
- [Edge AI Platforms](https://github.com/afondiel/Edge-AI-Platforms)
- [Edge AI Benchmarking](https://github.com/afondiel/Edge-AI-Benchmarking)
- [Edge AI Ecosystem](https://github.com/afondiel/computer-science-notebook/tree/master/core/systems/edge-computing/edge-ai/industry-applications)
- [Edge AI Books](https://github.com/afondiel/cs-books/blob/main/README.md#edge-computing)
- [Edge AI Blog](https://afondiel.github.io/posts/)
- [Edge AI Papers](https://github.com/afondiel/computer-science-notebook/tree/master/core/systems/edge-computing/edge-ai/resources/edge_ai_papers_news.md)

[Back to Table of Contents](#table-of-contents)
