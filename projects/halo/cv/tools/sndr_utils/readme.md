# sndr utils

## parse_ops_latency.py
- 作用：把snpe工具统计的每个op耗时汇总到csv中
- 用法：snpe工具的用法参考https://horizonrobotics.feishu.cn/wiki/wikcnHxKwycewBdRBFfQrYE1RXf
```
python3 parse_ops_latency.py --html_path 'html_file' --latency_path 'latency_file' --out_csv_path 'out_csv_path'
```
- html_path: snpe-dlc-viewer工具可视化dlc得到的html文件
- latency_path: snpe-diagview解析log得到的耗时（把打印到控制台的输出复制到txt文件中）
- out_csv_path: 期望输出的csv路径