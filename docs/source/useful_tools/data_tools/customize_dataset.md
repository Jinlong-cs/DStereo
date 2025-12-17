# 如何将自定义数据集转成COCO数据格式  

这篇文档介绍如何把自己的检测任务的数据集转成COCO数据集的格式，从而方便使用[COCODetectionMetric](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/blob/master/hat/metrics/coco_detection.py)接口做指标评测。  


## 数据格式的转换
这篇[文档](https://horizonrobotics.feishu.cn/docs/doccnIs8DtSosc83IL5BfjQqbUc)下分析了COCO数据集的标注文件所包含的关键key，主要有三个：images、annotations、categories。 
```python
'images': [
    {
        'file_name': '000000220310.jpg',
        'height': 500,
        'width': 333,
        'id': 220310,
    },
    ...
],

'annotations': [
    {
        'area': 702.1057499999998,
        'bbox': [473.07, 395.93, 38.65, 28.67],
        'category_id': 18,
        'id': 1768,
        'image_id': 289343,
        'iscrowd': 0,
        'segmentation': [[510.66,
            423.01,
            511.72,
            ...
            510.45,
            423.01]], # if you have mask labels
    },
    ...
],

'categories': [
    {'id': 1, 'name': 'person', 'supercategory': 'person'}
]
```  
每个key下面有些key是必不可少的。  
- images: images下是个list，每个list下的内容对应着一张图片的信息，需要提供的内容有file_name、height、width、id(图片的id，每个id对应一张图片)  
- annotations: annotations下是个list，每个list下的内容对应着一个标注框的相关信息，需要提供的内容有bbox(框的坐标信息，格式是xywh)、category_id(框的类别)、id(框的id，可以自己定义，不要重复即可)、image_id(图像的id，和images下面的id是一个东西)、iscrowd(框内是否有重合目标，默认情况设置为0即可)，area(框的面积)、segmentation(框的mask信息，默认情况下提供框的四个角的坐标即可)。  
- categories: categories下是个list，每个list下的内容对应着类别的id和类别的名称之间的关系，类别id可以不是连续的。  

当你需要把自己的数据格式转成COCO的数据格式的时候，只需要按照上面介绍的内容提供images、annotations、categories这三个关键的key，最后保存成json格式的文件即可。

## 注意事项
当你做完数据格式的转换之后，就可以利用COCODetectionMetric的接口来做自己的数据集的指标评测了，不过需要注意以下几点:  
- COCODetectionMetric要求你的模型最终输出的预测结果的格式是[x1,y1,x2,y2,score,label]的格式  
```python
pred_label = det[:, -1]
pred_score = det[:, -2]
pred_bbox = det[:, 0:4]
```
- COCODetectionMetric要求你的模型最终输出的预测结果是在预处理之后的图片上的结果，它会帮你resize到原始图片大小上
```python
pred_bbox = pred_bbox / output["scale_factor"][idx].cpu().numpy()
```
