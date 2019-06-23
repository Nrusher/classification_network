# 基于pytorch的分类网络框架

**HIT RNN_Control**

- 支持tensorboardx可视化
- 支持log文件输出
- 支持断点训练
- 支持保存最好模型

已支持分类网络结构如下：

- LeNet
- ResNet18
- ResNet34
- DesenNet
- VGG11
- VGG11bn
- VGG19bn
- EfficientNetb0
- AlexNet

**实验列表**

项目名_网络名_学习率变化方式_输入图像尺寸_batch大小_学习率 

|实验名|参数量|是否完成|
|:----------:|:--------------:|:--------------:|
|traffic_LeNet_224x224_16_0.01|1M|yes|
|traffic_LeNet_coslr_64x64_16_0.01|1M|yes|
|traffic_LeNet_coslr_64x64_64_0.01|1M|yes|
|traffic_LeNetcomplex_224x224_16_0.01|x|yes|
|traffic_LeNetcomplex_coslr_224x224_16_0.01|x|yes|
|traffic_LeNetcomplex_coslr_64x64_16_0.01|x|yes|
|traffic_ResNet18_coslr_64x64_64_0.01|11M|yes|
|traffic_ResNet18_coslr_64x64_16_0.01|11M|yes|
|traffic_ResNet34_coslr_64x64_16_0.01.log|21M|yes|
|traffic_EfficientNetb0_coslr_64x64_16_0.01|4M|yes|
|traffic_DesenNet_coslr_64x64_16_0.01|7M|yes|
|traffic_VGG19bn_coslr_64x64_16_0.01|190M|yes|
|traffic_VGG11_coslr_64x64_16_0.01|129M|yes|
|traffic_Alexnet_coslr_64x64_16_0.005.log|57M|yes|

输入图片尺寸影响（图像已平滑）

![input_size_val_loss](https://github.com/Nrusher/classification_network/blob/master/pic/input_size_val_loss.png)

![input_size_val_acc](https://github.com/Nrusher/classification_network/blob/master/pic/input_size_val_acc.png)

训练batch大小影响（图像已平滑）

![batch_size_val_loss](https://github.com/Nrusher/classification_network/blob/master/pic/batch_size_val_loss.png)

![batch_size_val_acc](https://github.com/Nrusher/classification_network/blob/master/pic/batch_size_val_acc.png)

**input_size = 64x64, batch_size = 16,初始学习率0.01情况下的网络实验对比图（图像已平滑）**

验证集误差对比

![val_loss](https://github.com/Nrusher/classification_network/blob/master/pic/val_loss.png)

验证集准确率对比

![val_acc](https://github.com/Nrusher/classification_network/blob/master/pic/val_acc.png)

学习率变化曲线

![learn_rate](https://github.com/Nrusher/classification_network/blob/master/pic/learn_rate.png)

最好模型准确率

|模型|准确率|参数量|
|:--:|:--:|:--:|
|VGG11|97.7%|129M|
|ResNet18|97.2%|11M|
|EfficientNetb0|97.2%|4M|
|DesenNet|96.8%|7M|
|LeNet|95.9%|1M|
|AlexNet|95.4%|57M|





