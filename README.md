# PyTorch CIFAR-10 图像分类项目

这是一个使用PyTorch构建的简单卷积神经网络（CNN），用于在CIFAR-10数据集上进行图像分类。

## 项目简介

本项目旨在演示如何使用PyTorch完成一个完整的深度学习图像分类任务，包括：
- 加载和预处理CIFAR-10数据集
- 构建一个简单的CNN模型
- 定义损失函数和优化器
- 训练模型
- 在测试集上评估模型性能

## 如何运行

1.  **克隆仓库**
    ```bash
    git clone https://github.com/kikoiio/pytorch-cifar10-cnn.git
    cd pytorch-cifar10-cnn
    ```

2.  **创建并激活Conda环境**
    ```bash
    # (确保你已经安装了Anaconda)
    conda create --name pytorch_env python=3.9
    conda activate pytorch_env
    ```

3.  **安装依赖**
    ```bash
    # (根据你之前安装PyTorch的命令来写，例如)
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

4.  **运行训练脚本**
    ```bash
    python train_cifar10.py
    ```

## 训练结果

经过2个周期的训练，模型在10000张测试图像上的准确率约为 XX%。（这里可以填上你运行脚本后得到的准确率）
