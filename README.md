# TRA Unofficial Reimplementation
> An Unofficial Reimplementation of [Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport (Lin et al., TRA, KDD2021)](https://dl.acm.org/doi/abs/10.1145/3447548.3467358?casa_token=BuvssX11egYAAAAA:ddUznyIdxvDN1wIJNzss-6T4__oJz3Vi_poOTGXgfUuA5HA4oooK6cbhS-xWLzyb8sQNCAxaHDk). 
> TRA 是基于 Microsoft QLib 框架组织的，数据模型内部处理不太透明，本项目抽丝剥茧，尽可能将其完全从 QLib 解耦出来，力图理清所有细节。

[\[Official Project\]](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/TRA/README.md)
[\[Official Dataset\]](https://drive.google.com/drive/folders/1fMqZYSeLyrHiWmVzygeI4sw3vp5Gt8cY)

**Notes**:

2022.11.19: TRA 的核心思想是捕捉股票特征中的多样的数据模式 (multi patterns), 其设计了多头预测器，并通过一种路由分配机制 (Temporal Routing Adaptor) 选择预测头。这跟我们最近的多样化人体动作预测的工作 [\[Diverse Human Motion Prediction via Gumbel-Softmax Sampling from an Auxiliary Space (Dang et al., ACMMM2022) \]](https://github.com/Droliven/diverse_sampling) 有异曲同工之妙，另外二者都涉及到了 gumbel-softmax 的采样机制。


## 本项目与原项目的区别

1. 剥离了 Microsoft QLib 相关代码，并保证没有造成任何影响。（我们承认 QLib 中有不少好用的功能，但是本项目中并没有用到多少）
2. 本项目在训练过程中直接根据测试集上的回测指标保存模型，这虽然是一种非法操作，但考录到直接运行原项目时，所得结果跟论文有不小的差距，因为用这种方式，可以找到模型输出指标最漂亮的极限
3. 本项目通过 tensorboard scalar 可以在训练过程中，监视各项指标的变化曲线，更加直观

## 运行

Download the dataset and modify data paths in the `configs` dictionary.

Take ALSTM as backbone:

1. `python main --config_path ./configs/config_alstm.yaml`
2. `python main --config_path ./configs/config_alstm_tra_init.yaml`
3. `python main --config_path ./configs/config_alstm_tra.yaml`

Take Transformer as backbone:

1. `python main --config_path ./configs/config_transformer.yaml`
2. `python main --config_path ./configs/config_transformer_tra_init.yaml`
3. `python main --config_path ./configs/config_transformer_tra.yaml`

## Todo

+ [x] 发现模型层、控制层是比较容易剥离的，比较麻烦的是数据层
+ [x] QLib.init() 这个函数没有也可以。其中传入的 `provider_uri`, `region` 并没有造成什么影响。
+ [x] tra 部分，测试的时候，gumbel softmax 加权求和的方式与训练的时候不同
+ [ ] 搞清楚 memory 机制，memory 写入的 index 是输入区间的右端点，但读出的时候，读的是左边 39 天的误差。另外在训练、验证、测试的过程中 memory 都一直在变，会不会造成影响。在训练过程中，memory.clear() 会造成什么影响？
+ [ ] 训练过程，数据的格式是 (B, 60, 16); 测试过程中，数据的格式是 (800, 60, 16), index 是怎么对上去的？
+ [ ] label 为什么是一个浮点数，代表什么？似乎是代表 ranking 的累计百分比
