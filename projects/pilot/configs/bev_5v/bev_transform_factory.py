# COPY FROM SD
# NOTE: this file records the customized tranform configurations for different camera modules # noqa
# Module details: http://wiki.hobot.cc/pages/viewpage.action?pageId=61621140 # noqa
# Input Shape:
#   Front:
#     0820, X8b: (2160, 3840)
#   Side:
#     weisen0233: (1280, 2048)
#     X3c: (1280, 1920)
#   Round:
#     0390: (1080, 1920)
#     isx031: (1536, 1920)
# Camera Module Type:
#   "0820_weisen0233",  # 6v default
#   "X8b_X3c",  # 6v
#   "0390",  # 4v default
#   "X8b_X3c_0390",  # 10v
#   "0820_weisen0233_isx031",  # 10v
#   "0820_weisen0233_isx031_0820",  # 11v
#   "X8b_X3c_isx031_X8b",  # 11v
# Change of Image Shape:
#   6v default:
#     front: (2160, 3840) --Resize--> (540, 960) --Crop--> (512, 960)
#     side: (1280, 2048) --Resize--> (640, 1024) --Crop--> (640, 1024)
#   X8b_X3c:
#     front: (2160, 3840) --Resize--> (540, 960) --Crop--> (512, 960)
#     side: (1280, 1920) --Resize--> (640, 960) --Crop--> (640, 960)
#   4v default:
#     round: (1080, 1920) --Resize--> (540, 960) --Crop--> (512, 960)
#   X8b_X3c_0390:
#     front: (2160, 3840) --Resize--> (540, 960) --Crop--> (512, 960)
#     side: (1280, 1920) --Resize--> (640, 960) --Crop--> (640, 960)
#     round: (1080, 1920) --Resize--> (540, 960) --Crop--> (512, 960)
#   0820_weisen0233_isx031:
#     front: (2160, 3840) --Resize--> (540, 960) --Crop--> (512, 960)
#     side: (1280, 2048) --Resize--> (640, 1024) --Crop--> (640, 1024)
#     round: (1536, 1920) --Resize--> (768, 960) --Crop--> (768, 960)
#   0820_weisen0233_isx031_0820:
#     front: (2160, 3840) --Resize--> (540, 960) --Crop--> (512, 960)
#     side: (1280, 2048) --Resize--> (640, 1024) --Crop--> (640, 1024)
#     round: (1536, 1920) --Resize--> (768, 960) --Crop--> (768, 960)
#     narrow: (2160, 3840) --Resize--> (540, 960) --Crop--> (512, 960)
#   X8b_X3c_isx031_X8b:
#     front: (2160, 3840) --Resize--> (540, 960) --Crop--> (512, 960)
#     side: (1280, 1920) --Resize--> (640, 960) --Crop--> (640, 960)
#     round: (1536, 1920) --Resize--> (768, 960) --Crop--> (768, 960)
#     narrow: (2160, 3840) --Resize--> (540, 960) --Crop--> (512, 960)
#   X8b_X3c_X8b:
#     front: (2160, 3840) --Resize--> (540, 960) --Crop--> (512, 960)
#     side: (1280, 1920) --Resize--> (640, 960) --Crop--> (640, 960)
#     narrow: (2160, 3840) --Resize--> (540, 960) --Crop--> (512, 960)

# NOTE: When the data collected by the new camera module needs to be added,
# we need to update the relevant transform parameters in this file.
#
# It's necessary to ensure that the transform parameters of the same module
# are the same. The transform parameters of all modules are resized according
# to the resolution in equal proportion and then obtained by crop.
#
# For the data collected by the same canera module, the flag_for_group is
# same. For the data collected by the new module, it needs to provide a
# new np.uint value for flag_for_group. All these notices are to cooperate
# with the group_sampler to ensure the shapes in the same batch are consistent.
#
# For example, when we need to add new camera module X8b_X3c_isx031, we notice
# that it is composed of 10v. It is different from the existing module type, so
# we need to add a new dict and set new value to the flag_for_group.
#
# Maybe you have a question, "X8b_X3c_isx031 is same as the 10v of X8b_X3c_isx031_X8b # noqa
# (total 11v), why we still need to set a new value as before?" In Fact, to support # noqa
# 11v mix-training, we could generate fake data for 4v/6v/10v dataset in 11v training. # noqa
# In default, the shape of fake data refers to default per_view_shape. In this case, # noqa
# narrow_view_shape of X8b is different from default value, so we need to set a new # noqa
# value to avoid fake narrow data and X8b appear simultaneously in the same batch. # noqa

# Default per_view_shape can be found in bev_base.py


H_PERSP_VIEW_SCALE = 1 / 4

# 6v
front0820_weisen0233 = {
    "per_view_shape": {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
    },
    "img_load_size": [(960, 540)] + [(1024, 640)] * 5,
    "transforms": {
        "ANCResize3DV": dict(
            type="ANCResize3DV",
            size=[(540, 960)] + [(640, 1024)] * 5,
        ),
        "ANCCrop3DV": dict(
            type="ANCCrop3DV",
            height=[512] + [640] * 5,
            width=[960] + [1024] * 5,
            top=[0] + [0] * 5,
            left=[0] + [0] * 5,
        ),
    },
    "flag_for_group": 1,
}

# 6v
X8b_X3c = {
    "per_view_shape": {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 1920),
        "camera_front_right": (1280, 1920),
        "camera_rear_left": (1280, 1920),
        "camera_rear_right": (1280, 1920),
        "camera_rear": (1280, 1920),
    },
    "img_load_size": [(1920, 1080)] + [(960, 640)] * 5,
    "transforms": {
        "ANCResize3DV": dict(
            type="ANCResize3DV",
            size=[(540, 960)] + [(640, 960)] * 5,
        ),
        "ANCCrop3DV": dict(
            type="ANCCrop3DV",
            height=[512] + [640] * 5,
            width=[960] + [960] * 5,
            top=[0] + [0] * 5,
            left=[0] + [0] * 5,
        ),
    },
    "flag_for_group": 2,
}

# 10v
X8b_X3c_0390 = {
    "per_view_shape": {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 1920),
        "camera_front_right": (1280, 1920),
        "camera_rear_left": (1280, 1920),
        "camera_rear_right": (1280, 1920),
        "camera_rear": (1280, 1920),
        "fisheye_front": (1080, 1920),
        "fisheye_rear": (1080, 1920),
        "fisheye_left": (1080, 1920),
        "fisheye_right": (1080, 1920),
    },
    "img_load_size": [(1920, 1080)] + [(960, 640)] * 5 + [(960, 540)] * 4,
    "transforms": {
        "ANCResize3DV": dict(
            type="ANCResize3DV",
            size=[(540, 960)] + [(640, 960)] * 5 + [(540, 960)] * 4,
        ),
        "ANCCrop3DV": dict(
            type="ANCCrop3DV",
            height=[512] + [640] * 5 + [512] * 4,
            width=[960] + [960] * 5 + [960] * 4,
            top=[0] + [0] * 5 + [28] * 4,
            left=[0] + [0] * 5 + [0] * 4,
        ),
    },
    "flag_for_group": 2,
}

front0820_weisen0233_isx031 = {
    "per_view_shape": {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
        "fisheye_front": (1536, 1920),
        "fisheye_rear": (1536, 1920),
        "fisheye_left": (1536, 1920),
        "fisheye_right": (1536, 1920),
    },
    "img_load_size": [(960, 540)] + [(1024, 640)] * 5 + [(960, 768)] * 4,
    "transforms": {
        "ANCResize3DV": dict(
            type="ANCResize3DV",
            size=[(540, 960)] + [(640, 1024)] * 5 + [(768, 960)] * 4,
        ),
        "ANCCrop3DV": dict(
            type="ANCCrop3DV",
            height=[512] + [640] * 5 + [768] * 4,
            width=[960] + [1024] * 5 + [960] * 4,
            top=[0] + [0] * 5 + [0] * 4,
            left=[0] + [0] * 5 + [0] * 4,
        ),
    },
    "flag_for_group": 3,
}

# 4v
isx031 = {
    "per_view_shape": {
        "fisheye_front": (1536, 1920),
        "fisheye_rear": (1536, 1920),
        "fisheye_left": (1536, 1920),
        "fisheye_right": (1536, 1920),
    },
    "img_load_size": [(960, 768)] * 4,
    "transforms": {
        "ANCResize3DV": dict(
            type="ANCResize3DV",
            size=[(768, 960)] * 4,
        ),
        "ANCCrop3DV": dict(
            type="ANCCrop3DV",
            height=[768] * 4,
            width=[960] * 4,
            top=[0] * 4,
            left=[0] * 4,
        ),
    },
    "flag_for_group": 3,
}

# 11v
front0820_weisen0233_isx031_0820 = {
    "per_view_shape": {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
        "fisheye_front": (1536, 1920),
        "fisheye_rear": (1536, 1920),
        "fisheye_left": (1536, 1920),
        "fisheye_right": (1536, 1920),
        "camera_front_30fov": (2160, 3840),
    },
    "img_load_size": [(1920, 1080)]
    + [(1024, 640)] * 5
    + [(960, 768)] * 4
    + [(1920, 1080)],
    "transforms": {
        "ANCResize3DV": dict(
            type="ANCResize3DV",
            size=[(540, 960)]
            + [(640, 1024)] * 5
            + [(768, 960)] * 4
            + [(540, 960)],
        ),
        "ANCCrop3DV": dict(
            type="ANCCrop3DV",
            height=[512] + [640] * 5 + [768] * 4 + [512],
            width=[960] + [1024] * 5 + [960] * 4 + [960],
            top=[0] + [0] * 5 + [0] * 4 + [0],
            left=[0] + [0] * 5 + [0] * 4 + [0],
        ),
    },
    "flag_for_group": 3,
}

X8b_X3c_isx031_X8b = {
    "per_view_shape": {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 1920),
        "camera_front_right": (1280, 1920),
        "camera_rear_left": (1280, 1920),
        "camera_rear_right": (1280, 1920),
        "camera_rear": (1280, 1920),
        "fisheye_front": (1536, 1920),
        "fisheye_rear": (1536, 1920),
        "fisheye_left": (1536, 1920),
        "fisheye_right": (1536, 1920),
        "camera_front_30fov": (2160, 3840),
    },
    "img_load_size": [(1920, 1080)]
    + [(960, 640)] * 5
    + [(960, 768)] * 4
    + [(1920, 1080)],
    "transforms": {
        "ANCResize3DV": dict(
            type="ANCResize3DV",
            size=[(540, 960)]
            + [(640, 960)] * 5
            + [(768, 960)] * 4
            + [(540, 960)],
        ),
        "ANCCrop3DV": dict(
            type="ANCCrop3DV",
            height=[512] + [640] * 5 + [768] * 4 + [512],
            width=[960] + [960] * 5 + [960] * 4 + [960],
            top=[0] + [0] * 5 + [0] * 4 + [0],
            left=[0] + [0] * 5 + [0] * 4 + [0],
        ),
    },
    "flag_for_group": 4,
}

# 5v
X3c = {
    "per_view_shape": {
        "camera_front_left": (1280, 1920),
        "camera_front_right": (1280, 1920),
        "camera_rear_left": (1280, 1920),
        "camera_rear_right": (1280, 1920),
        "camera_rear": (1280, 1920),
    },
    "img_load_size": [(960, 640)] * 5,
    "transforms": {
        "ANCResize3DV": dict(type="ANCResize3DV", size=[(640, 960)] * 5),
        "ANCCrop3DV": dict(
            type="ANCCrop3DV",
            height=[640] * 5,
            width=[960] * 5,
            top=[0] * 5,
            left=[0] * 5,
        ),
    },
    "flag_for_group": 5,
}

# 7v
X8b_X3c_X8b = {
    "per_view_shape": {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 1920),
        "camera_front_right": (1280, 1920),
        "camera_rear_left": (1280, 1920),
        "camera_rear_right": (1280, 1920),
        "camera_rear": (1280, 1920),
        "camera_front_30fov": (2160, 3840),
    },
    "img_load_size": [(1920, 1080)] + [(960, 640)] * 5 + [(1920, 1080)],
    "transforms": {
        "ANCResize3DV": dict(
            type="ANCResize3DV",
            size=[(540, 960)] + [(640, 960)] * 5 + [(540, 960)],
        ),
        "ANCCrop3DV": dict(
            type="ANCCrop3DV",
            height=[512] + [640] * 5 + [512],
            width=[960] + [960] * 5 + [960],
            top=[0] + [0] * 5 + [0],
            left=[0] + [0] * 5 + [0],
        ),
    },
    "flag_for_group": 5,
}

weisen0233 = {
    "per_view_shape": {
        "camera_front_left": (1280, 2048),
        "camera_front_right": (1280, 2048),
        "camera_rear_left": (1280, 2048),
        "camera_rear_right": (1280, 2048),
        "camera_rear": (1280, 2048),
    },
    "img_load_size": [(1024, 640)] * 5,
    "transforms": {
        "ANCResize3DV": dict(
            type="ANCResize3DV",
            size=[(640, 1024)] * 5,
        ),
        "ANCCrop3DV": dict(
            type="ANCCrop3DV",
            height=[640] * 5,
            width=[1024] * 5,
            top=[0] * 5,
            left=[0] * 5,
        ),
    },
    "flag_for_group": 1,
}

X8b_X3c_isx031_X8b_EK = {
    "per_view_shape": {
        "camera_front": (2160, 3840),
        "camera_front_left": (1280, 1920),
        "camera_front_right": (1280, 1920),
        "camera_rear_left": (1280, 1920),
        "camera_rear_right": (1280, 1920),
        "camera_rear": (1280, 1920),
        "fisheye_front": (1280, 1920),
        "fisheye_rear": (1280, 1920),
        "fisheye_left": (1280, 1920),
        "fisheye_right": (1280, 1920),
        "camera_front_30fov": (2160, 3840),
    },
    "img_load_size": [(1920, 1080)]
    + [(1024, 640)] * 5
    + [(960, 640)] * 4
    + [(1920, 1080)],
    "transforms": {
        "ANCResize3DV": dict(
            type="ANCResize3DV",
            size=[(540, 960)]
            + [(640, 960)] * 5
            + [(640, 960)] * 4
            + [(540, 960)],
        ),
        "ANCCrop3DV": dict(
            type="ANCCrop3DV",
            height=[512] + [640] * 5 + [640] * 4 + [512],
            width=[960] + [960] * 5 + [960] * 4 + [960],
            top=[0] + [0] * 5 + [0] * 4 + [0],
            left=[0] + [0] * 5 + [0] * 4 + [0],
        ),
    },
    "flag_for_group": 6,
}
