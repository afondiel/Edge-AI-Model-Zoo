[![](https://img.shields.io/badge/Contribute-Welcome-green)](./CONTRIBUTING.md)

# Edge-AI Model Zoo

A curated list of pre-trained, ready-to-deploy models optimized for edge devices, sourced from AI hubs like Hugging Face, GitHub, TensorFlow Hub, PyTorch Hub, and more.

## Table of Contents
- [Finding the Perfect Edge AI Model for Your Application](#finding-the-perfect-edge-ai-model-for-your-application)
- [Model Zoos](#model-zoos)
- [Real-World Uses for Edge AI Models](#real-world-uses-for-edge-ai-models)
- [Resources](#resources)

## Finding the Perfect Edge AI Model for Your Application
Selecting the right model for edge deployment is critical for balancing performance, efficiency, and resource constraints. Here’s a quick guide for edge AI developers, practitioners, and industry experts:

### Why It Matters
- **Efficiency**: Edge devices (e.g., IoT, mobile, embedded systems) have limited compute, memory, and power.
- **Performance**: Real-time applications (e.g., autonomous drones, smart cameras) demand low latency and high accuracy.
- **Scalability**: The right model ensures cost-effective deployment across devices.

### Key Criteria
1. **Task Requirements**: Match the model to your application (e.g., vision, audio, multimodal).
2. **Hardware Constraints**: Consider compute (OPS), memory (MB), and energy (mWh) limits of your device.
3. **Performance Goals**: Balance accuracy, latency, and throughput for your use case.
4. **Deployment Ease**: Check compatibility with frameworks (e.g., TensorFlow Lite, ONNX).

**Next Steps**: Once you’ve shortlisted a model, use the [Model Benchmarking, Profiling and Optimization Guide](./model-bench-prof-opt-guide.md) to profile and optimize it.

## Model Zoos

[Back to Table of Contents](#table-of-contents)

| Model Zoo | Description | Links |
|------------------------|---------------------------------------------------------------------|----------------------------------------------------|
| Edge AI Labs Model Zoo | A collection of pre-trained, optimized models for low-power devices.| [EdgeAI Labs](https://edgeai.modelnova.ai/models/) |
| Edge Impulse Model Zoo | A repository of models optimized for edge devices. | [Edge Impulse Model Zoo](https://www.edgeimpulse.com/) |
| ONNX Model Zoo | A collection of pre-trained, state-of-the-art models in the ONNX format. | [ONNX Model Zoo](https://github.com/onnx/models) |
| NVIDIA Pretrained AI Models (NGC + TAO)| Accelerate AI development with world-class customizable pretrained models from NVIDIA. | - [NVIDIA Pretrained AI Models - Main](https://developer.nvidia.com/ai-models) <br> - [NGC Model Catalog](https://catalog.ngc.nvidia.com/models?filters=&orderBy=weightPopularDESC&query=&page=&pageSize=) <br> -  [TAO Model Zoo](https://docs.nvidia.com/tao/tao-toolkit/text/model_zoo/overview.html)|
| OpenVINO Model Zoo | A collection of pre-trained models ready for use with Intel's OpenVINO toolkit. | [OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) |
| Qualcomm Models Zoo | A collection of AI models from Qualcomm. | [Qualcomm Models Zoo](https://github.com/quic/ai-hub-models/) |
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

## Real-World Uses for Edge AI Models

[Back to Table of Contents](#table-of-contents)

The table below categorizes some of these models based on their primary capabilities for real-world applications:

| Category | Model | Description | Reference |
|---|---|---|---|
| **Language** | Whisper | General-purpose speech recognition model | [Whisper on Hugging Face](https://huggingface.co/openai/whisper) |
| | Baichuan | Large language model | [Baichuan on Hugging Face](https://huggingface.co/baichuan-inc/Baichuan-13B-Base) |
| | huggingface_wavlm_base_plus | Audio language model for speech recognition | [WavLM on Hugging Face](https://huggingface.co/microsoft/wavlm-base-plus) |
| | whisper_asr | Speech recognition model | [Whisper ASR on GitHub](https://github.com/openai/whisper) |
| | trocr | Text recognition in images | [TrOCR on Hugging Face](https://huggingface.co/microsoft/trocr-base) |
| |MobileLLM | Large language model | [MobileLLM on Hugging Face](https://huggingface.co/collections/facebook/mobilellm-6722be18cb86c20ebe113e95)|
| **Audio** | Whisper | Speech-to-text | [Whisper on Hugging Face](https://huggingface.co/openai/whisper) |
| | huggingface_wavlm_base_plus | Audio language model for speech recognition | [WavLM on Hugging Face](https://huggingface.co/microsoft/wavlm-base-plus) |
| | whisper_asr | Speech recognition model | [Whisper ASR on GitHub](https://github.com/openai/whisper) |
| **Vision** | aotgan | Image generation | [AOT-GAN on GitHub](https://github.com/researchmm/AOT-GAN) |
| | convnext_tiny | Vision transformer for image classification | [ConvNeXt on Hugging Face](https://huggingface.co/facebook/convnext-tiny-224) |
| | ddrnet23_slim | Image segmentation | [DDRNet on GitHub](https://github.com/ydhongHIT/DDRNet) |
| | deeplabv3_resnet50 | Semantic image segmentation | [DeepLabV3 on TensorFlow Hub](https://tfhub.dev/tensorflow/deeplabv3/1) |
| | densenet121 | Image classification | [DenseNet on PyTorch Hub](https://pytorch.org/hub/pytorch_vision_densenet/) |
| | detr_resnet101 | Object detection | [DETR on GitHub](https://github.com/facebookresearch/detr) |
| | detr_resnet101_dc5 | Object detection | [DETR on GitHub](https://github.com/facebookresearch/detr) |
| | detr_resnet50 | Object detection | [DETR on GitHub](https://github.com/facebookresearch/detr) |
| | detr_resnet50_dc5 | Object detection | [DETR on GitHub](https://github.com/facebookresearch/detr) |
| | esrgan | Image super-resolution | [ESRGAN on GitHub](https://github.com/xinntao/ESRGAN) |
| | facebook_denoiser | Image denoising | [Denoiser on GitHub](https://github.com/facebookresearch/denoiser) |
| | fcn_resnet50 | Semantic image segmentation | [FCN on PyTorch Hub](https://pytorch.org/hub/pytorch_vision_fcn_resnet50/) |
| | googlenet | Image classification | [GoogLeNet on PyTorch Hub](https://pytorch.org/hub/pytorch_vision_googlenet/) |
| | lama_dilated | Image inpainting | [LaMa on GitHub](https://github.com/saic-mdal/lama) |
| | litehrnet | Image segmentation | [Lite-HRNet on GitHub](https://github.com/HRNet/Lite-HRNet) |
| | mediapipe_face | Face detection | [MediaPipe Face on Google AI](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html) |
| | mediapipe_hand | Hand detection | [MediaPipe Hand on Google AI](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html) |
| | mnasnet05 | Image classification | [MnasNet on PyTorch Hub](https://pytorch.org/hub/pytorch_vision_mnasnet/) |
| | mobiledet | Object detection | [MobileDet on TensorFlow Hub](https://tfhub.dev/s?q=mobiledet) |
| | mobilenet_v3_small | Image classification | [MobileNetV3 on TensorFlow Hub](https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5) |
| | openpose | Human pose estimation | [OpenPose on GitHub](https://github.com/CMU-Perceptual-Computing-Lab/openpose) |
| | QuickSRNet_Large | Image super-resolution | [QuickSRNet on GitHub](https://github.com/QuickSRNet/QuickSRNet) |
| | real_esrgan_general_x4v3 | Image super-resolution | [Real-ESRGAN on GitHub](https://github.com/xinntao/Real-ESRGAN) |
| | real_esrgan_x4plus | Image super-resolution | [Real-ESRGAN on GitHub](https://github.com/xinntao/Real-ESRGAN) |
| | resnet_2plus1d | Video classification | [ResNet-2+1D on PyTorch Hub](https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/) |
| | resnet_3d | Video classification | [ResNet-3D on PyTorch Hub](https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/) |
| | ResNeXt50 | Image classification | [ResNeXt on PyTorch Hub](https://pytorch.org/hub/pytorch_vision_resnext/) |
| | sam | Segmentation | [SAM on GitHub](https://github.com/facebookresearch/segment-anything) |
| | Sesr_m3 | Image super-resolution | [SESR on GitHub](https://github.com/SESR/SESR) |
| | shufflenet_v2 | Image classification | [ShuffleNetV2 on PyTorch Hub](https://pytorch.org/hub/pytorch_vision_shufflenet_v2/) |
| | sinet | Image segmentation | [SINet on GitHub](https://github.com/DengPingFan/SINet) |
| | squeezenet_1 | Image classification | [SqueezeNet on PyTorch Hub](https://pytorch.org/hub/pytorch_vision_squeezenet/) |
| | stylegan2 | Image generation | [StyleGAN2 on GitHub](https://github.com/NVlabs/stylegan2) |
| | unet_segmentation | Image segmentation | [UNet on GitHub](https://github.com/milesial/Pytorch-UNet) |
| | vit | Image classification | [ViT on Hugging Face](https://huggingface.co/google/vit-base-patch16-224) |
| | wideresnet50 | Image classification | [WideResNet on PyTorch Hub](https://pytorch.org/hub/pytorch_vision_wideresnet/) |
| | xisr | Image super-resolution | [XISR on GitHub](https://github.com/XISR/XISR) |
| | yolov6 | Object detection | [YOLOv6 on GitHub](https://github.com/meituan/YOLOv6) |
| | yolov7 | Object detection | [YOLOv7 on GitHub](https://github.com/WongKinYiu/yolov7) |
| | yolov8_det | Object detection | [YOLOv8 on GitHub](https://github.com/ultralytics/yolov8) |
| | yolov8_seg | Object detection and segmentation | [YOLOv8 on GitHub](https://github.com/ultralytics/yolov8) |
| **Multimodality** | ControlNet | Fine control over image generation | [ControlNet on GitHub](https://github.com/lllyasviel/ControlNet) |
| | Stable Diffusion | Text-to-image generation | [Stable Diffusion on Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4) |
| | Mediapipe_pose | Human pose estimation | [MediaPipe Pose on Google AI](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html) |
| **Other** | pinecone | Vector database | Pinecone [invalid URL removed] |
| | weaviate-c2 | Vector database | Weaviate [invalid URL removed] |
| | Intel | Various models | Intel AI [invalid URL removed] |
| | kakao-enterprise | Various models | Hugging Face [invalid URL removed] |
| | laion | Various models | Hugging Face [invalid URL removed] |
| | openai | Various models | OpenAI [invalid URL removed] |
| | runwayml | Various models | Hugging Face [invalid URL removed] |
| | Salesforce | Various models | Hugging Face [invalid URL removed] |
| | stabilityai | Various models | Hugging Face [invalid URL removed] |
| | upstage | Various models | Hugging Face [invalid URL removed] |
| | ybelkada | Various models | Hugging Face [invalid URL removed] |

## Resources

- [Edge AI Engineering](https://github.com/afondiel/edge-ai-engineering) 
- [Edge Vision](https://github.com/afondiel/edge-vision)
- [Edge AI Platforms](https://github.com/afondiel/Edge-AI-Platforms)
- [Edge AI Model Zoo](https://github.com/afondiel/Edge-AI-Model-Zoo)
- [Edge AI Frameworks](https://github.com/afondiel/computer-science-notebook/tree/master/core/systems/edge-computing/edge-ai/frameworks)
- [Edge AI Deployment Stack](https://github.com/afondiel/computer-science-notebook/tree/master/core/systems/edge-computing/edge-ai/concepts/deployment)
- [Edge AI Qualcomm Stack](https://www.qualcomm.com/developer/artificial-intelligence)
- [Edge AI Benchmarking & Optimization](https://github.com/afondiel/Edge-AI-Model-Zoo/blob/main/model-bench-prof-opt-guide.md)
- [Edge AI Technical Guide](https://github.com/afondiel/computer-science-notebook/tree/master/core/systems/edge-computing/edge-ai/concepts)
- [Edge AI Ecosystem](https://github.com/afondiel/computer-science-notebook/tree/master/core/systems/edge-computing/edge-ai/industry-applications)
- [Edge AI Books](https://github.com/afondiel/cs-books/blob/main/README.md#edge-computing)
- [Edge AI Blog](https://afondiel.github.io/posts/)

[Back to Table of Contents](#table-of-contents)
