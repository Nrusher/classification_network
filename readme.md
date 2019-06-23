# 基于pytorch的分类网络框架

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



**实验列表**

项目名_网络名_学习率变化方式_输入图像尺寸_batch大小_学习率 

|实验名|是否完成|
|:----------:|:--------------:|
|traffic_LeNet_224x224_16_0.01||
|traffic_LeNetcomplex_224x224_16_0.01|yes|
|traffic_LeNetcomplex_coslr_224x224_16_0.01|yes|
|traffic_LeNetcomplex_coslr_64x64_16_0.01|yes|
|traffic_LeNet_coslr_64x64_16_0.01|yes|
|traffic_LeNet_coslr_64x64_64_0.01|yes|
|traffic_ResNet18_coslr_64x64_64_0.01|yes|
|traffic_ResNet18_coslr_64x64_16_0.01|yes|
|traffic_ResNet18_coslr_64x64_16_0.01|yes|
|traffic_EfficientNetb0_coslr_64x64_16_0.01|yes|
|traffic_DesenNet_coslr_64x64_16_0.01|yes|
|traffic_VGG19bn_coslr_64x64_16_0.01|yes|
|traffic_VGG11_coslr_64x64_16_0.01|yes|


