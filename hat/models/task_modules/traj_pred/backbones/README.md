增加这个文件的原因是，网络结构的设计中，有很多地方有相对复杂的设计，或是与原版模型有一些差别。但是我的洋屁水平不足以支持我表示清楚上面的内容，而中文注释受限于sphinx的一些奇葩规则，又常常通不过CI，因此我将这些地方叙述在本文档中。

# SGNet
SGNet的论文和源代码解读见：https://horizonrobotics.feishu.cn/docs/doccnRelJckLwwmy9t7aCu24a0b。
我们在迁移SGNet的过程中，参照源代码进行了很大的改动和适配，具体改动如下：

## Encoder
1. 原模型使用的是GRU，我们改成了LSTM。

2. 我们强行将原文的模型结构斩成两半，区分出了encoder和decoder。原模型的循环次数过多，在LSTM中的复杂度为O(enc_steps * dec_steps ^ 2)，运算的时延过大，经我们修改，复杂度变为了O(enc_steps * dec_steps)。这里的复杂度指的是LSTM循环的复杂度。

2.1 原模型O(enc_steps * dec_steps ^ 2)的复杂度，是enc_lstm每一步，都会循环dec_steps步，生成长度为dec_steps的goal_traj（以及相应的goal_h），在训练的时候，goal_traj与gt会产生一个loss。而每一步的goal_traj，又产生了dec_steps步的循环，生成当前时刻~dec_steps步之后的预测轨迹。就像滑窗一样，其实中间生成的很多“未来的轨迹”并不会被真正使用，而是白生成了。

2.2 鉴于2.1，我们只需要在enc_lstm的最后一步（enc_steps - 1）产生预测即可，输入只依赖enc_steps - 1步骤的goal_h。因此我们在这里可以将模型一分为二，产生了encoder和decoder。

3. SGE net部分，原文中实际上就是想做attention，但是实现的与attention有差别，我们就直接替换成self-attention了。

## Decoder
1. 原模型在cvae模块后面，也是先做enc steps的大循环，再在下边函数`cvae_decoder_forward`做dec steps次的小循环。我觉得没什么必要。
