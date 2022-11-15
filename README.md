# TRA Reimplementation
> TRA (KDD2021) 是基于 QLib 调度框架的，数据模型内部处理不太透明，本项目抽丝剥茧，尽可能将其完全从 QLib 解耦出来，力图理清所有细节。

## Todo

+ [ ] 发现模型层、控制层是比较容易剥离的，比较麻烦的是数据层
+ [ ] QLib.init() 这个函数没有也可以。其中传入的 `provider_uri`, `region` 并没有造成什么影响。
+ [ ] Dataset -> slice 不同公司不同天数之间是怎么对齐的，还没有理清楚
+ [ ] 搞清楚 memory 机制
