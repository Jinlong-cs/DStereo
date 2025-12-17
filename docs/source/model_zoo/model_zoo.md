# ModelZoo

## Classification

| network | float | qat | quantization | dataset | input shape | bpu latency (ms) |
| :---- | :---: | :---: | :---: | :---: | :---: | :---: |
| MobileNetV1 | 74.12 | 73.92 | 73.61 | ImageNet | 1x3x224x224 | 0.77 |
| MobileNetV2 | 72.65 | 72.51 | 72.11 | ImageNet | 1x3x224x224 | 0.69 |
| ResNet 18 | 72.04 | 72.03 | 72.03 | ImageNet | 1x3x224x224 | 1.53 |
| ResNet 50 | 77.37 | 76.99 | 76.94  | ImageNet | 1x3x224x224 | 3.06 |
| VargNetV2 | 73.94 | 73.56 | 73.64 | ImageNet | 1x3x224x224 | 0.78 |
| EfficientNet-B0 | 74.31 | 74.23 | 74.18 | ImageNet | 1x3x224x224 | 0.91 |
| SwinTransformer | 80.24 | 80.15 | 80.05 | ImageNet | 1x3x224x224 | 14.50 |
| MixVarGENet | 71.33 | 71.23 | 71.04 | ImageNet | 1x3x224x224 | 0.56 |

## Detection

**FCOS**

| network | backbone | float | qat | quantization | dataset | input shape | bpu latency (ms) |
| :---- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FCOS-efficientnet | efficientnetb0 | 36.26 | 35.79 | 35.59 |MS COCO| 1x3x512x512 | 1.35 |


更多的参考模型可在 [工具链参考模型] (http://j5.ddk.hobot.cc/cn/hat/source/examples.html) 中获取。