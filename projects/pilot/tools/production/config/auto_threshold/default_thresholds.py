from copy import deepcopy

_default_threshold = dict(
    rpn_thresh=dict(
        person=0.40,
        cyclist=0.45,
        vehicle=0.3,
        rear=0.3,
    ),
    det_thresh=dict(
        person=0.1,
        cyclist=0.1,
        vehicle=0.1,
        rear=0.1,
    ),
    roi_det_thresh=dict(
        vehicle=0.3,
        rear=0.1,
        person=0.1,
    ),
    thresh_3d=dict(
        person=0.1,
        cyclist=0.1,
        vehicle=0.1,
    ),
    bbox_clipping=dict(
        person=False,
        cyclist=False,
        vehicle=False,
        rear=False,
    ),
)

update_fn = lambda old_d, new_d: old_d.update(new_d) or old_d

default_pilot_model_threshold = {
    ("niofy_cn_x3c_rear_night", "crop_bayes"): update_fn(
        deepcopy(_default_threshold),
        dict(
            rpn_thresh=dict(
                person=0.40,
                cyclist=0.25,
                vehicle=0.3,
                rear=0.3,
            )
        ),
    ),
    ("niofy_cn_x3c_rear_day", "crop_bayes"): update_fn(
        deepcopy(_default_threshold),
        dict(
            rpn_thresh=dict(
                person=0.40,
                cyclist=0.25,
                vehicle=0.3,
                rear=0.3,
            )
        ),
    ),
}
