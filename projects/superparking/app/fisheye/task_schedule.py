from importlib import import_module

from common import int_infer_stage, is_int_infer, val_only

add_2nd_stage = True
tasks_1st_stage = dict(
    vehicle_tasks=[
        dict(name="vehicle_detection", important=True),
        dict(name="vehicle_detection_3d", important=True, compile=False),
    ],
    person_tasks=[
        dict(name="person_detection", important=True),
        dict(name="person_detection_3d", important=True, compile=False),
    ],
    cyclist_tasks=[
        dict(name="cyclist_detection", important=True),
        dict(name="cyclist_detection_3d", important=True, compile=False),
    ],
    rear_tasks=[
        dict(name="rear_detection", important=True),
    ],
    sod_tasks=[
        dict(name="sod", important=True),
    ],
    parsing_tasks=[
        dict(name="parsing", important=True),
    ],
)

tasks_2nd_stage = dict(
    vehicle_tasks=[
        dict(name="vehicle_category_classification"),
        dict(name="vehicle_occlusion_classification"),
        dict(name="vehicle_truncation_classification"),
        dict(name="vehicle_wheel_kps", important=True),
    ],
    rear_tasks=[
        dict(name="rear_plate_detection"),
        dict(name="rear_occlusion_classification"),
        dict(name="rear_part_classification"),
    ],
    person_tasks=[
        dict(name="person_face_detection"),
        dict(name="person_occlusion_classification"),
        dict(name="person_orientation_classification"),
        dict(name="person_pose_classification"),
    ],
)

all_tasks = tasks_1st_stage
if is_int_infer:
    if val_only:
        pass
    elif int_infer_stage == "stage_one":
        add_2nd_stage = False
    elif int_infer_stage == "stage_two":
        all_tasks = dict(
            vehicle_tasks=[], person_tasks=[], cyclist_tasks=[], rear_tasks=[]
        )

if add_2nd_stage:
    for task_name in tasks_2nd_stage:
        if task_name not in tasks_1st_stage:
            continue
        all_tasks[task_name].extend(tasks_2nd_stage[task_name])

tasks = []
for task_group in all_tasks.values():
    tasks.extend(task_group)

task_names = []
for t in tasks:
    obj_type = t["name"].split("_")[0]
    task_name = f".task_cfg.{obj_type}.{t['name']}"
    if is_int_infer and not val_only and t.get("compile", True) is False:
        continue
    task_names.append(task_name)

TASK_CONFIGS = [
    import_module(t, "projects.superparking.app.fisheye") for t in task_names
]
