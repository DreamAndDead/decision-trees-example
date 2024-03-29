# triple threat offence

这个数据集是关于篮球三威胁进攻，对于进攻使用有效策略的数据集。

其中
- offence_all.xlsx，是所有已知数据
- offence_train.xlsx，训练数据
- offence_test.xlsx，测试数据

三威胁是一个进攻者，在非内线区域接球后所使用的进攻先导姿势。
在这个姿势下，可以根据情况，进行
- 传球
- 投篮
- 突破

这个数据集描述了面对不同的防守者，如何使用有效的行动。

## columns

假如自己是一名进攻者，此时在外围区域接到了传球，面对篮框，摆出了三威胁姿势。

### basket distance

这一列表示自己到篮框的距离，距离越近，表明自己的投篮命中率越高。
所以相应的值是由自己的投篮能力决定的。
- close，投篮很有把握
- moderate，处于自己投篮范围的极限，必要情况下也可以出手
- far，太远了，没什么把握投进

### op distance

对于一对一的防守来说，总有一个防守者在尝试限制你。
这一列表明他到你的距离，这个距离是由他是否可以限制你的投篮来决定的。
- close，他离你很近，强行投篮很可能会被封盖/干扰
- moderate，刚好处于可以封盖的边缘，如果进行出手，会有心理上的干扰
- far，很远，对投篮没有过大的威胁

### op movement

在每个时刻，防守者都会有一些行动，比如他觉得距离过远，想要向前靠近；或者想要后退，防止你的突破。
这一列表明防守者的移动情况。
- forward，防守者主动向前靠近
- still，防守者保持原地不动
- back，防守者主动向后后退

### target

这一列表明自己在看到眼前的情况时，可以采取的最理想的行动。
- drive，进行突破
- drive fake，突破虚晃
- shoot，投篮
- shoot fake，投篮虚晃
- pass，传球（样本中没有这个结果，数据好像来自比较独的人）

