import json
import os
import re
import argparse

# 定义日志目录
log_dir = "output/hat_logs/"
# 定义输入和输出文件路径
input_file_path = "input/task.json"
output_file_path = "output/result.json"

def create_output_directory():
    """确保输出目录存在"""
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

def load_task_config():
    """读取任务配置文件"""
    with open(input_file_path, 'r') as input_file:
        return json.load(input_file)

def find_latest_log_file(category):
    """找到最近的日志文件"""
    latest_file = None
    latest_time = 0

    for filename in os.listdir(log_dir):
        if category == 'train' and filename.startswith("train") and "rank" not in filename:
            file_path = os.path.join(log_dir, filename)
            file_time = os.path.getmtime(file_path)  # 获取文件的修改时间
            
            # 更新最近的文件
            if file_time > latest_time:
                latest_time = file_time
                latest_file = file_path

    return latest_file

def extract_training_metrics(latest_file):
    """从最近的日志文件提取训练指标"""
    training_results = {"metrics": {}}

    if latest_file:
        with open(latest_file, 'r') as log_file:
            lines = log_file.readlines()  # 读取所有行
            
            # 从最后一行开始向上查找
            for line in reversed(lines):
                if "train_DStereoTask" in line:
                    # 使用正则表达式匹配 loss 和 EPE
                    match = re.search(r'loss\[(\d+\.\d+)\].*EPE\[(\d+\.\d+)\]', line)
                    if match:
                        training_results["metrics"]["loss"] = float(match.group(1))
                        training_results["metrics"]["epe"] = float(match.group(2))
                        break  # 找到后可以退出循环

    return training_results

def extract_quantization_metrics(model_name):
    """提取量化指标"""
    quant_info_path = f"output/ptq_result/{model_name}_quant_info.json"

    with open(quant_info_path, "r") as f:
        quant_info = json.load(f)

    cosine_similarities = {}
    for k, v in quant_info.items():
        if "cosine_similarity" in v:
            cosine_similarities[k] = float(v["cosine_similarity"])

    return {
        "metrics": cosine_similarities,
        "fallback": True  # 根据需要设置fallback的值
    }

def main():
    """主函数"""
    create_output_directory()
    task_config = load_task_config()

    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='获取训练或量化指标')
    parser.add_argument('-c', '--category', choices=['train', 'quant'], required=True, 
                        help='选择指标类型: train（训练指标）或 quant（量化指标）')
    args = parser.parse_args()

    # 提取训练指标
    result_data = {}
    model_info = task_config.get("model", {})
    if args.category == 'train':
        latest_file = find_latest_log_file('train')
        training_results = extract_training_metrics(latest_file)
        result_data["training_results"] = training_results
        model_info.update({
            "weights_path": "output/DStereoTask/float-checkpoint-last.pth.tar",
            "export_path": "output/float.onnx",
        })
    else:
        quantization_results = extract_quantization_metrics("DStereoTask")
        result_data["quantization_results"] = quantization_results
        model_info.update({"bin_path": "output/ptq_result/DStereoTask.bin"})


    # 创建要写入的结果字典  
    result_data["model_info"] = model_info

    # 将结果写入 output/result.json 文件
    with open(output_file_path, 'w') as output_file:
        json.dump(result_data, output_file, indent=4)

    print(f"metrics result insert {output_file_path}")

if __name__ == "__main__":
    main()