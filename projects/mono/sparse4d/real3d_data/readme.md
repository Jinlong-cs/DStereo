# sparse4d 单帧模型
- 使用real3d数据集
- 关键点投影考虑畸变
- 3v输入:resize + crop*2

#### 注意

`use_deformable_func`为True情况下记得先build cuda op 具体可以执行以下命令

```bash
cd hat/models/task_modules/sparse4d/ops
python3 setup.py develop
```
