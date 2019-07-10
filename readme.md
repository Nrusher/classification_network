# 基于卷积神经网络的交通标志分类

**HIT ANN_Control**

## 1.源码功能特点

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

## 2.文件结构

├─data(用来保存数据集)

|     ├─train(训练集数据)

|     ├─val(验证集数据)

| ├─test(测试集数据)

├─logs(用来保存训练日志)

├─model(自己搭建的网络结构)

├─model_save(用来断点模型和最优模型)

├─pic(用来保存相关图片)

analysis.py : 用来分析对比网络效果

mian.py : 训练网络主函数

mian.py : 分割数据集，统计样本分布等基础功能函数

## 3.依赖环境及运行方法

**依赖**
python版本：python3.7

CUDA版本：CUDA:10.0

依赖库：
pytorch 1.1
efficientnet_pytorch
tensorboardX
time
logging
以及常用库

**运行方法**
在main.c文件中修改以下类的默认参数，配置实验使用的网络及其参数（请确保数据已按目录分割好）
```python
class Arg():
    def __init__(self,
                 project_name='test', # 工程名
                 class_num=62, # 类别数
                 input_size=(64, 64), # 输入尺寸
                 lr=0.01, # 初始学习率
                 epoch=100, # 训练轮数
                 cuda='cuda', # 使用GPU(cuda)还是CPU(cpu)
                 train_root='../traffic/data/train', # 训练数据目录
                 train_batch_size=16, # 训练batch大小
                 val_root='../traffic/data/val', # 验证集目录
                 val_batch_size=16, # 验证batch大小
                 load='make_model', # make_model：创建新模型，load_params：加载参数继续上次训练，load_model:加载整个模型
                 model_type='ResNet18', # 使用的网络结构名
                 model_save_dir='./model_save', # 模型保持目录
                 model_load_dir='./model_save/traffic_DesenNet_224x224_16.ckp.params.pth', # 参数加载文件模型文件
                 log_dir='./logs', # 日志保存目录
                 save_mode='save_params', # 保存参数还是整个模型
                 checkpoint_per_epoch=5, # 每几轮保存一次
                 using_tensorboardx=True, # 是否使用可视化功能
                 tensorboardx_file='./logs', # tensorboardx文件保存目录
                 verbose=1 # 打印方式，1或0
                 ):
```

## 4.实验列表及结果

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