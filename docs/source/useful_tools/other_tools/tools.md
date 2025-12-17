# 如何使用tools

HAT中提供了丰富的tools，例如模型计算量计算、模型编译、训练、预测等工具。

例如计算量计算工具，可以通过以下命令使用：

```bash
python3 tools/calops.py --config ${CONFIG_PATH}

```

同时，HAT的whl包中还集成了训练、预测、模型检查、计算量计算、模型编译、提交集群工具。

## 安装HAT的whl包后可以通过以下方式使用：

### 计算量计算工具:

```bash
python3 -m hat.cli.calops --config ${CONFIG_PATH}
```

### 模型检查

```bash
python3 -m hat.cli.model_checker --config ${CONFIG_PATH}

```

### 训练工具：

```bash
python3 -m hat.cli.train --stage float --config ${CONFIG_PATH}

```

### 预测工具：

```bash
python3 -m hat.cli.predict --stage float --config ${CONFIG_PATH}

```

### 编译工具:

```bash
python3 tools/deploy/compile_perf.py --config ${CONFIG_PATH}
```

```bash
python3 -m hat.cli.compile_standalone ${PT_FILE} --input-size ${INPUT_SIZE} --name ${NAME} --opt O2 --march ${MARCH}

```
