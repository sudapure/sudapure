# TensorRT Release of Person ReID UseCase based on ResNet-IBN-18-a Architecture
This release contains baseline TensorRT version of IBN-Net(ResNet-18-a/ResNet-18-b) backbone architecture to solve person Re-ID, M.A.R problem as covered in PR #66, this implementation primarily contains TensorRT based deployment release for research release [v1.4-beta](https://github.com/onearmbandit/VAS-Core-Research/releases/tag/v1.4-beta), this release was supposed to be published on 21/03/2022 but due to PR reviews & other priority tasks it got delayed, for detailed varying combination of baseline research experimentation's for this implementation carried in release [v1.4-beta](https://github.com/onearmbandit/VAS-Core-Research/releases/tag/v1.4-beta) and documented [here](https://docs.google.com/document/d/1OMxwlCedrsxOHhfWJDdXBUI44zozGDlpwGGIOMNAa_A/edit#heading=h.rwubogtek0go).

## Change Logs
* Pre-Processing and Normalization using TensorRT by @sudapure in https://github.com/onearmbandit/VAS-Video-Analytics/pull/56
* Upgradation of the Core to Tensorrt 8.2.x by @ckhire in https://github.com/onearmbandit/VAS-Video-Analytics/pull/61
* Upgradation by @ckhire in https://github.com/onearmbandit/VAS-Video-Analytics/pull/67
* TRT Network implementation of IBN-Net architecture for Re-ID/M.A.R use-case by @sudapure in https://github.com/onearmbandit/VAS-Video-Analytics/pull/66

## Issues Resolved
- Exporting the PyTorch trained weights to intermediate key/value pair representation :heavy_check_mark: :1st_place_medal: 
- DVC integration for versioning all weight/engine files. :heavy_check_mark: :1st_place_medal: 
- Building the pre-trained weight specific Network architecture using TRT based Network Builder API. :heavy_check_mark: :1st_place_medal: 
- Implementing the pre-processing transform as a part of custom TRT layer . :heavy_check_mark: :1st_place_medal: 
- Implementation of basic non-optimal inference module in TensorRT C++. :heavy_check_mark: :1st_place_medal: 
- Implementing the use-case specific post-processing function. :heavy_check_mark: :1st_place_medal: 
- cross-validating the accuracy benchmarks of TensorRT deployed model with the PyTorch implementation. :heavy_check_mark: :1st_place_medal: 

## Specification
<table>
<tr>
<th>Metric</th>
<th>Value</th>
</tr>
<tr>
<td>Market-1501 rank@1 accuracy</td>
<td>84</td>
</tr>
<tr>
<td>Market-1501 mAP</td>
<td>0.67</td>
</tr>
<tr>
<td>Pose coverage</td>
<td>Standing upright, parallel to image plane</td>
</tr>
<tr>
<td>Camera Angle</td>
<td><= &ang;40&#176;</td>
</tr>
<tr>
<td>Support of occluded pedestrians </td>
<td>Yes</td>
</tr>
<tr>
<td>Occlusion coverage  </td>
<td> <50% </td>
</tr>
<tr>
<td>GFlops  </td>
<td>N/A </td>
</tr>
<tr>
<td>MParams   </td>
<td>N/A </td>
</tr>
<tr>
<td>Source framework </td>
<td>PyTorch </td>
</tr>
</table>

## Implementation
**IBN-Net**, is a novel convolutional architecture, which remarkably enhances a CNN?s modeling ability on one domain (e.g. Cityscapes) as well as its generalization capacity on another domain (e.g. GTA5) without fine-tuning. IBN-Net carefully integrates Instance Normalization (IN) and Batch Normalization (BN) as building blocks, and can be wrapped into many advanced deep networks to improve their performances, as per the experimentation's conducted down the line IBN has been incorporated with ResNet-18 architecture, it can also be integrated with other S.O.T.A architectures like OS-Net, Efficient-Net, Res-Next, or even use-case specific custom CNN's.

### Network Architecture
<p align='center'>
<img src="https://user-images.githubusercontent.com/57352045/148694823-025b41dc-4759-447a-8d4b-87c89ea3a164.png"/>
</p>

Model was trained from random weight initialization for 120 epoch and a `mAP` score of 0.6767 was achieved on 119th epoch. All Training logs experiments stored in DVC. Below results are taken with TTA(Test Time Augmentation) however it is observed that without TTA accuracy drops drastically,
<p align='center'>
<img src="https://user-images.githubusercontent.com/57352045/157899672-2fd4ecde-b248-4b85-8cf5-c06abe8093fa.png"/>
</p>

As primary objective of this release implementation is to achieve similar output tensor precision as that of PyTorch based implementation, output tensor after inferring the input using PyTorch model in Python,
```
tensor([[ -1.4042,  -3.8353,  -2.2664,   0.6956,  -0.2306,  -6.5067,  -6.0886,
          -6.0619,  -8.1355,  -3.5969,  -6.7499,  -1.3107,  -1.7550,   0.2188,
          -1.2123,  -3.6320,  -7.5967, -19.1849, -11.9088, -14.6207, -15.7484,
         -13.2949,  -1.2161]], device='cuda:0',
       grad_fn=<CudnnBatchNormBackward>)
```
output tensor after inferring the same input using TRT model in C++,
```
-1.40416 -3.83534 -2.26642 0.695616 -0.230602 -6.50673 -6.08859 -6.06186 -8.13554 -3.59686 
-6.74991 -1.31073 -1.75503 0.218828 -1.21227 -3.63196 -7.59674 -19.1848 -11.9088 -14.6207 
-15.7484 -13.2949 -1.2161
```
As from above output tensors they are almost identical with minor precision difference, this confirms that the converted model from PyTorch to TRT is producing same results.

<p align='center'>
<img height="2400" src="https://user-images.githubusercontent.com/57352045/161441449-e5ec91d9-0cbe-4541-8bee-1599865a0928.png"/>
</p>

### Design Architecture
Below diagram shows different components and it's respective dimensions used in training phase and corresponding output nodes that are used for training and inferencing, here the ReID head will change in dimensions for Person & Vehicle use-case, however the backbone network will remain same for both the use-cases.
<p align='center'>
<img src="https://user-images.githubusercontent.com/57352045/157907477-5eeb8e5e-5bcd-46b6-872e-3a195aa6ee98.png"/>
</p>

### Application Utility
For quick evaluation  of trained Re-ID model the feature embedding are used with image retrieval problem to retrieve relevant appearing images from the set of gallery images, below are two scenarios where in first one the query image & retrieved results are close enough as in the second case the input & retrieved images are pretty distinct indicating as worst case scenario.
<table>
<th colspan="2" align='center'>
Best Case
</th>
<tr align='center'>
<td>Input Query</td>
<td>Top-k Results</td>
</tr>

<tr>
<td><img src="https://user-images.githubusercontent.com/57352045/158007533-4895d3f7-2451-4b50-9678-e85fa0c7bc6c.png"/></td>
<td><img src="https://user-images.githubusercontent.com/57352045/158007531-8292a70e-f732-42d0-8e9d-ce40413e4458.png"/></td>
</tr>

<th colspan="2" align='center'>
Worst Case
</th>
<tr align='center'>
<td>Input Query</td>
<td>Top-k Results</td>
</tr>

<tr>
<td><img src="https://user-images.githubusercontent.com/57352045/158019941-26ace696-dd50-4bb4-888c-d6cc1d799c02.png"/></td>
<td><img src="https://user-images.githubusercontent.com/57352045/158019938-0a93d7d9-92df-49e7-be6a-3fa035ce938e.png"/></td>
</tr>
</table>

## Conclusive Remark
The primary purpose of this release is to export the `IBN-Net-Resnet-18-a` architecture to intermediate format, and validating the accuracy benchmarks with respect to the baseline implementation as already implemented in PyTorch, as done in [this](https://github.com/onearmbandit/VAS-Core-Research/pull/54) PR. The expected outcome of this release is getting the near similar benchmark number for accuracy by optimizing the infer latency using TensorRT specific implementation for as that of the research based PyTorch implementation as documented [here](https://docs.google.com/document/d/1rnSRnbJQLRgfR3DjcOuHTVHA51f_Q8tPZpK0H-t5ep0/edit#heading=h.of82xekw4fe).
The infer latency has been emphasized for this release as a part of deployment phase,  the accuracy numbers are subject to training & research phase as covered in release [v1.4-beta](https://github.com/onearmbandit/VAS-Core-Research/releases/tag/v1.4-beta) onearmbandit/VAS-Core-Research.

## known Issues
- The code has not been refactored as per preliminary design pattern of S.O.L.I.D convention guidelines (#TODO), refer https://github.com/onearmbandit/VAS-Core-Research/discussions/31.
- Without TTA accuracy drops drastically, this is very critical issue as in deployment release TTA is not required.
- Model is not trained with Reviewed Dataset, also single source dataset is used for training & validation.

## Further improvements (#TODO):
- Improving accuracy without TTA, which is crucial for deployment specific releases.
- NN architecture optimization & decreasing trainable parameter to further improve latency numbers
- Implementation of custom NN operators to optimize infer latency by keeping accuracy intact.
- Implementation of PTQ(Post Training Quantization) or QAT(Quantization Aware Training) for optimizing infer latency using INT8 Quantization mode.

## Evaluation Metrics
<table>
<tr>
<td>Version</td>
<td>Use-Case</td>
<td>Framework/IE</td>
<td>Architecture(GPU/CPU)</td>
<td>Algorithm</td>
<td>Precision</td>
<td>Accuracy</td>
<td>latency</td>
<td>FPS</td>
</tr>

<tr>
<td>v1.1-beta</td>
<td>Person Re-ID</td>
<td>TensorRT</td>
<td>Nvidia GTX 1060</td>
<td>IBN-Net-ResNet-18-a</td>
<td>FP32</td>
<td>0.65 mAP</td>
<td>1.70 ms</td>
<td>N/A</td>
</tr>
<tr>
<td>v1.1-beta</td>
<td>Person Retrieval</td>
<td>TensorRT</td>
<td>Nvidia GTX 1060</td>
<td>IBN-Net-ResNet-18-a</td>
<td>FP32</td>
<td>84% @ Rank-1</td>
<td>1.70 ms</td>
<td>N/A</td>
</tr>

</table>

** Accuracy here we referring to is based on standard CMC `mAP` and ranking based evaluation metrics with TTA, however the `mAP` used in this case is different from the `mAP` parameter used in detection or segmentation use-case. Also the achieved accuracy numbers are near identical to the one observed in research release [v1.4-beta](https://github.com/onearmbandit/VAS-Core-Research/releases/tag/v1.4-beta).

** Latency<sub>GPU/CPU</sub> measures mentioned in the experiments are considered with respect to backbone model inference, the end-to-end latency numbers for image retrieval has not been considered as it is altogether a subject to different experimentation and use-case specific requirement.. 

