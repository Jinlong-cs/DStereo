# 如何使用裁剪工具

## 裁剪工具介绍

HAT面向不同的项目，需要提供一套裁剪工具，以便于面向不同的项目做交付。
目前已有的项目toolchain，fsd，pilot，根据项目需要的内容，对代码库进行裁剪发布。

## 快速使用

代码入口为`plugins/code_stripping/code_stripping.py`，需要到`plugins/code_stripping`目录下执行：

```shell
    python3 code_stripping.py --file-list toolchain-file-list.py --target-dir ../../release/HAT --override
```
`code_stripping.py` 一共有以下几个参数：

`--file-list`: 配置文件的路径,是必须要传入的参数。

`--src-dir`: 源地址，默认为`../../`。

`--target-dir`: 为目标地址，默认为`./release`。

`--override`: 表示是否覆盖， 默认为`False`。

`--float-crop`: 参数表示是否是浮点裁剪，这个是浮点裁剪包需要的，含plugin的裁剪不需要这个参数，默认为`False`。

`--clip-code`: 参数表示是否裁剪文件内部代码，详细使用方式可见下面介绍，默认为`False`。

## 配置文件的书写格式

配置文件支持yaml格式和py格式，推荐使用py格式。
配置文件中`file_list`、`skip_file_list`、`copytodst`、`init_dirs`需要关注，若不使用的话无需设置，其中`init_dirs`有默认值。

### file_list
`file_list`中是需要保留的文件夹或文件路径，支持glob库支持的格式，例如`**`，`*`等通配符，如下示例：
```python
    file_list = [
        "hat/__init__.py",
        "hat/callbacks/callbacks.py",
        "hat/callbacks/checkpoint.py",
        "hat/callbacks/save_eval_results",
        "hat/callbacks/task_visualize/",
        "hat/models/task_modules/lidar_encoder/*",
        "hat/**/traj_pred_*",
    ]
```
### skip_file_list

`skip_file_list` 中是需要跳过的文件夹或文件路径，支持的格式和`file_list`一样。
程序会在 file_list 拷贝完的基础上，根据 `skip_file_list` 中的匹配情况，在`--target-dir` 对应的位置删除对应的文件(或文件夹)以及 `__init__.py`（`__init__.py` 后面会根据裁剪的结果重写）
具体使用场景为：在`file_list`中写通配符，然后在`skip_file_list`将不需要的跳过，也可以在合并多个`config`中使用（见下面）。

```python
skip_file_list = [
    "hat/models/task_modules/yolo/head.py",
    "hat/data/samplers/dist_cycle_sampler_multi_dataset.py",
    "tests/unit_tests/models/base_modules/test_activation.py",
    "hat/models/base_modules/postprocess",
]
```

### copytodst

`copytodst`中为一些特殊需求使用的接口，可以将一个事先更改了的文件去覆盖或拷贝到目标位置。它是一个字典，每个`key`是在`--src-dir`目录下的相对路径，对应的`value`是在`--target-dir`下的相对路径。
```python
    copytodst = {
        "projects/float_toolchain/hat/*": "hat/",
        "projects/float_toolchain/thirdparty": "hat/thirdparty",
    }
```

如上代码，会将`--src-dir`目录中`projects/float_toolchain/hat/*`下的文件去覆盖`--target-dir`目录中的`hat/`下对应的文件，其严格按顺序执行`cp -rf {key} {value}`。

### init_dirs 

`init_dirs` 中是需要重写`__init__.py`的目录，可以定义为列表也可以定义为字典。
如果未定义，则默认为:
```python
    init_dirs = {
        "hat": "hat",
        "tests": "tests",
    }
```

1、若裁剪前和裁剪后需要重写__init__.py的文件夹在根目录下的路径相同，则可以定义为:
```python
    init_dirs = ["hat", "tests"]
```
上面的示例会在程序中将列表解析为一个字典，其`key`和`value`相同，为列表中的一个元素。

每个`key`是在`--src-dir`目录下的相对路径，其对应的`value`是在`--target-dir下`的相对路径。程序会递归遍历`key`目录下的所有子目录，并根据裁剪结果在其`value`对应的路径下生成新的`__init__.py`（若在裁剪后的目录中存在`__init__.py`，则会跳过生成，比如`hat/__init__.py`）。

2、若裁剪前和裁剪后需要重写`__init__.py`的文件夹在根目录下的路径不相同，则可以使用字典来定义。例如，我们将`--src-dir`目录下的`projects/float_toolchain/hat/callbacks`这个目录裁剪后拷贝到了`--target-dir`下的`hat/callbacks`中，那么可以使用如下定义来重写剪裁后`hat/callbacks`中的`__init__.py`文件：

```python
    init_dirs = {
    "projects/float_toolchain/hat/callbacks": "hat/callbacks",
    }
```
一般来讲，只需要使用默认值就可以，即不用在配置文件中定义。

### 多个config合并

当使用py格式的config时，可以将多个config合并，然后解析出你需要的以上`key`的值就行了，例如：
```python
    from importlib import import_module
    list1 = [
        "debug_1",
        "debug_2"
    ]

    old_configs = [import_module(t) for t in list1]
    """
    ..............
    """
    file_list = xxx
    init_dirs = xxx
    skip_file_list = xxx
    copytodst = xxx
```

## 浮点裁剪

由于某些项目发版不需要`horizon-plugin-pytorch`、`aidisdk`和`hatbc`，因此需要将这些包的依赖去掉。目前使用的方法是:
1、伪造三个名字相同的包，并实现该项目中需要用到的方法，在裁剪时，将这些包放到`hat/thirdparty`中，并修改所有的`import`，让其从`hat.thirdparty`中去`import`。具体的实现参考[dev-toolchain分支](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/tree/dev-toolchain)。
2、通过缩进匹配的方式，在文件中删除`set_qconfig`、`fuse_model`、`set_calibration_qconfig`函数，建议构建模型时将这些函数独自用到的`import`写到函数体中，好处在于该import可以随着函数的删除也一并删除了。


## 裁剪文件内部代码

可以通过使用`--clip-code`参数，来实现文件内部的裁剪代码。具体用法为在需要裁剪的代码上方添加`# clipping: begin`, 下方添加 `# clipping: end`。如下所示：
```python
    x1 = 1
    # clipping: begin
    x2 = 2
    # clipping: end
    x3 = 3
```

裁剪后文件中的代码如下：
```python
    x1 = 1
    x3 = 3
```