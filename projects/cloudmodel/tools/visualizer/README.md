# 可视化工具
这里是一个云端模型的可视化工具。 目前面向的数据结构是 hatbc中的hatbc.message.frame.CameraFrame.

目前支持的感知结构包括:
1. 目标检测(Instance)
2. 语义分割(DenseMap)
3. 车道线实例分割(Instance)
4. 图片/ROI 分类 (Instance)

# 快速上手
使用上很简单，用户只需要提供正确的 CameraFrame结构即可完成渲染。  
下面的例子同时可视化了检测、语义分割、实例分割结果。
```python
import os
import uuid
import pickle
import torch
from hatbc.message import (
    BBox2D,
    CameraFrame,
    Image,
    Instance,
    Attribute,
    DenseMap,
    Mask2D,
)
from projects.cloudmodel.tools.visualizer.render_camera_frame import RenderCameraFrame
from projects.cloudmodel.tools.visualizer.class_metas.cloudmodel.class_meta_cloudmodel import (
    ClassMetaCloudModel,
)

img_url = "dmpv2://auto_image_2/adas_data_raw/pack/white_changan/LX179_20220526_D/ADAS_20220526-065400_215_6/ADAS_20220526-065400_215_6__59175_1653519326001_0_extra.jpg"
render_perceptions = list()
# --------------------------  detection with attribute ------------------------------
bbox = BBox2D(
    data=[2829.9057, 647.3422241210938, 3834.11865234375, 2160],
    topic="vehicle",
    score=0.5886231,
)
attributes = [
    Attribute(topic="part", value="full", score=1.0),
    Attribute(topic="occlusion", value="occluded", score=0.95),
    Attribute(topic="ignore", value="no", score=1.0),
    Attribute(topic="category", value="BigTruck", score=0.998),
    Attribute(topic="confidence", value="High", score=0.6836),
]
vehicle = Instance(
    topic="vehicle_detection",
    bbox2ds=[bbox],
    attributes=attributes
)
render_perceptions.append(vehicle)
# ----------------------- parsing densemap --------------------------
with open(
    "/horizon-bucket/auto_jenkins_test/test_cloudmodel/test_render/densemap_remder_demo_results.pkl", "rb"
) as f:
    mask = pickle.load(f)
mask = Mask2D(data=mask, topic="semantic_parsing_40cls_segmentation")
parsing_densemap = DenseMap(topic="semantic_parsing_40cls_segmentation", mask2ds=[mask])
render_perceptions.append(parsing_densemap)
# ----------------------- lane parsing mask --------------------------
lane_attributes = [
    Attribute(meta=None, value="Road_teeth", topic="lane_type", score=0.605),
    Attribute(meta=None, value="no", topic="lane_double_line", score=0.9999),
    Attribute(meta=None, value="other", topic="lane_color", score=0.9972),
    Attribute(meta=None, value="full_visible", topic="lane_occlusion", score=0.781),
]
with open(
    "/horizon-bucket/auto_jenkins_test/test_cloudmodel/test_render/laneinstancemask_render_demo_results.pkl",
    "rb",
) as f:
    mask = torch.tensor(pickle.load(f))
mask = Mask2D(data=mask, topic="lane_instanceseg")
instance_mask = Instance(
    mask2ds=[mask], attributes=lane_attributes
)
render_perceptions.append(instance_mask)

frame = CameraFrame(
    image=Image(url=img_url),
    perceptions=render_perceptions,
)

# ----------------------- render results --------------------------
visualizer = RenderCameraFrame(image_scale=2.0)
class_meta = ClassMetaCloudModel()
render_res = visualizer(frame, metadata=class_meta)
save_dir = os.path.join("tmp", f"{uuid.uuid4().hex}.jpg")
os.makedirs(os.path.dirname(save_dir), exist_ok=True)
render_res.save(save_dir)
```
可视化效果:
![render_result](http://image-viewer.auto-algorithm.hobot.cc/filesystem/bucket/output/auto_jenkins_test/test_cloudmodel/test_render/render_demo_result.jpg)