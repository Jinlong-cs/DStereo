import os

def move_stereo_files(base_path, input_file):
    # 创建目标文件夹
    left_dir = os.path.join(base_path, 'left')
    right_dir = os.path.join(base_path, 'right')
    disp_dir = os.path.join(base_path, 'disp')

    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    os.makedirs(disp_dir, exist_ok=True)

    # 读取输入文件
    with open(input_file, 'r') as f:
        for line in f:
            # 分割每一行，获取左目、右目和视差图的路径
            left_img, right_img, disp_img = line.strip().split()

            # 构建源文件的完整路径
            left_src = os.path.join('/mnt/data/StereoDepthDatasets', left_img)
            right_src = os.path.join('/mnt/data/StereoDepthDatasets', right_img)
            disp_src = os.path.join('/mnt/data/StereoDepthDatasets', disp_img)

            # 获取文件名
            left_filename = os.path.basename(left_img)
            right_filename = os.path.basename(right_img)
            disp_filename = os.path.basename(disp_img)

            # 使用juicefs clone命令克隆文件到目标文件夹
            os.system(f"juicefs clone {left_src} {os.path.join(left_dir, left_filename)}")
            os.system(f"juicefs clone {right_src} {os.path.join(right_dir, right_filename)}")
            os.system(f"juicefs clone {disp_src} {os.path.join(disp_dir, disp_filename)}")

            print(f"Cloned: {left_src} to {os.path.join(left_dir, left_filename)}")
            print(f"Cloned: {right_src} to {os.path.join(right_dir, right_filename)}")
            print(f"Cloned: {disp_src} to {os.path.join(disp_dir, disp_filename)}")

if __name__ == "__main__":
    base_path = '/mnt/data/StereoDepthDatasetsFromat/'  # 目标路径
    input_file = 'input/sceneflow_val.txt'  # 输入文件路径
    move_stereo_files(base_path, input_file)