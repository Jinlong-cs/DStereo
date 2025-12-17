model_id = "id107"
cfg_file = f"projects/mono/veh_side_mul3d_{model_id}/multitask.py"


pretrained_ckpt = "http://fm-jin-yang.bcloud-2nd.hogpu.cc/plat_gpu/mono-veh_side_mul3d_id101-20230401-140355/output/models/vehicle_side/qat-checkpoint-last.pth.tar"
pretrained_ckpt = "http://fm-jin-yang.bcloud-2nd.hogpu.cc/plat_gpu/mono-veh_side_mul3d_id102-20230401-151538/output/models/vehicle_side/qat-checkpoint-last.pth.tar"
pretrained_ckpt = "http://fm-jin-yang.bcloud-2nd.hogpu.cc/plat_gpu/mono-veh_side_mul3d_id103-20230401-150823/output/models/vehicle_side/qat-checkpoint-last.pth.tar"
pretrained_ckpt = "http://fm-jin-yang.train.hogpu.cc/plat_gpu/mono-veh_side_mul3d_id104-20230403-192258/output/models/vehicle_side/qat-checkpoint-last.pth.tar"
pretrained_ckpt = "http://fm-jin-yang.bcloud-2nd.hogpu.cc/plat_gpu/mono-veh_side_mul3d_id106-20230404-181802/output/models/vehicle_side/qat-checkpoint-last.pth.tar"
pretrained_ckpt = "http://fm-jin-yang.tcloud.hogpu.cc/plat_gpu/mono-veh_side_mul3d_id107_resume-20230410-121448/output/models/vehicle_side/qat-checkpoint-last.pth.tar"
assert model_id in pretrained_ckpt

# merge_face_plate = False
merge_face_plate = True
