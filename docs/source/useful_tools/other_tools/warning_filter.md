# 过滤 warning

目前由于海图的训练 log 中 warning 过多，尤其是在多机训练的时候过多的 warning 掩盖了其他有用的信息，因此海图添加了 warning 过滤的功能。用户可以在 config 文件中通过设置 filter_warning=True 来达到过滤每个 GPU 所唯一对应的训练启动进程中所有 warning 的目的。

受限于 HAT 训练时会启动子进程来读取数据的现状和 logging 设置不能跨进程生效，如果读取数据的子进程引发了 warning 打印，这部分 warning 将无法被过滤，如 Dataset 的初始化和获取数据中产生的 warning, 以及启动子进程时复制主进程环境导入了第三方库引发的 warning.
此外，如果是 C/C++ 层面的 warning, 本机制无法过滤。

**请注意**

由于 warning 过滤也可能带来信息遗漏，产生负面效果，在你打开 warning 过滤功能后，同时会产生 warning 被过滤的提醒信息。
