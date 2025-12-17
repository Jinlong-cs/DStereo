该目录下为分析工具， 其中：

`compare_dump_data.py`:  数值一致性对比工具，在环境或模型发生变化时，对 “变化是否会导致训练产生差异” 进行初步检查，帮助发现升级可能会带来的风险等。[数值一致性检查工具](http://model.aidi.hobot.cc/api/docs/HAT/latest/html/tutorials/common_tools/diff_dump_data.html)。

`model_profiler.py`: 量化训练debug工具，可以帮助用户来分析定位量化模型结构是否符合预期、模型精度损失具体发生在哪一层等。[量化训练Debug工具](http://model.aidi.hobot.cc/api/docs/HAT/latest/html/tutorials/common_tools/model_profiler.html)。

`perf_dataloader.py`：数据读取速度测试工具, 使用方式如下：
    ```bash
    python tools/perf_dataloader.py --config ${cfg} --stage float --allow-all-rank
    ```

`perf_model_training.py`：模型训练速度测试工具, 使用方式如下：
    ```bash
    python tools/perf_model_training.py --config ${cfg} --stage float --allow-all-rank
    ```

`task_similarity.py`：任务相似度分析工具，主要用来指导多任务分组训练和辅助任务训练。 [多任务相似度分析工具](http://model.aidi.hobot.cc/api/docs/HAT/latest/html/tutorials/common_tools/task_similarity.html)。

