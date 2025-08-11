#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tqdm进度条库完整演示项目
基于博客：https://blog.csdn.net/AI_dataloads/article/details/134169038

本项目演示tqdm的各种用法：
1. 基础进度条使用
2. 进度条描述设置
3. write方法使用
4. 深度学习中的应用
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm, trange
import os


def demo_basic_usage():
    """演示基础用法"""
    print("=" * 50)
    print("1. 基础进度条演示")
    print("=" * 50)

    print("\n1.1 传入可迭代对象:")
    for i in tqdm(range(100), desc="基础进度条"):
        time.sleep(0.02)  # 减少等待时间以便演示

    print("\n1.2 使用trange:")
    for i in trange(100, desc="trange演示"):
        time.sleep(0.02)


def demo_description():
    """演示进度条描述设置"""
    print("\n" + "=" * 50)
    print("2. 进度条描述设置演示")
    print("=" * 50)

    items = ["数据预处理", "模型训练", "模型验证", "结果保存"]
    pbar = tqdm(items)
    for item in pbar:
        pbar.set_description(f"正在处理: {item}")
        time.sleep(1)


def demo_write_method():
    """演示tqdm的write方法"""
    print("\n" + "=" * 50)
    print("3. tqdm write方法演示")
    print("=" * 50)

    bar = trange(20, desc="任务处理")
    for i in bar:
        time.sleep(0.1)
        if not (i % 5):
            tqdm.write(f"✓ 完成任务 {i}")


class SimpleNeuralNetwork(nn.Module):
    """简化的神经网络模型"""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(28 * 28, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.out(x)
        return x


def demo_deep_learning():
    """演示在深度学习中的应用"""
    print("\n" + "=" * 50)
    print("4. 深度学习中的tqdm应用演示")
    print("=" * 50)

    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建简化的数据集（避免下载大文件）
    print("\n4.1 创建模拟数据集...")
    # 创建模拟的MNIST数据
    train_data = torch.randn(1000, 1, 28, 28)  # 1000个样本
    train_labels = torch.randint(0, 10, (1000,))  # 对应标签

    test_data = torch.randn(200, 1, 28, 28)  # 200个测试样本
    test_labels = torch.randint(0, 10, (200,))

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 创建模型
    print("\n4.2 创建神经网络模型...")
    model = SimpleNeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练函数
    def train_epoch(dataloader, model, loss_fn, optimizer):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # 使用tqdm显示训练进度
        pbar = tqdm(dataloader, desc="训练中")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)

            # 前向传播
            outputs = model(data)
            loss = loss_fn(outputs, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条描述
            accuracy = 100. * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%'
            })

        return total_loss / len(dataloader), 100. * correct / total

    # 测试函数
    def test_epoch(dataloader, model, loss_fn):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="测试中")
            for data, targets in pbar:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                test_loss += loss_fn(outputs, targets).item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # 更新进度条
                accuracy = 100. * correct / total
                pbar.set_postfix({'Acc': f'{accuracy:.2f}%'})

        return test_loss / len(dataloader), 100. * correct / total

    # 开始训练
    print("\n4.3 开始训练模型...")
    epochs = 3

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)

        # 训练
        train_loss, train_acc = train_epoch(train_dataloader, model, loss_fn, optimizer)

        # 测试
        test_loss, test_acc = test_epoch(test_dataloader, model, loss_fn)

        # 输出结果
        tqdm.write(f"Epoch {epoch + 1} 结果:")
        tqdm.write(f"  训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        tqdm.write(f"  测试 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")


def demo_advanced_features():
    """演示tqdm的高级功能"""
    print("\n" + "=" * 50)
    print("5. tqdm高级功能演示")
    print("=" * 50)

    print("\n5.1 嵌套进度条:")
    for i in trange(3, desc="外层循环"):
        for j in trange(50, desc=f"内层循环 {i + 1}", leave=False):
            time.sleep(0.01)

    print("\n5.2 手动更新进度条:")
    pbar = tqdm(total=100, desc="手动更新")
    for i in range(10):
        # 模拟不规则的进度更新
        progress = (i + 1) * 10
        pbar.update(10)
        pbar.set_postfix({"步骤": f"{i + 1}/10"})
        time.sleep(0.2)
    pbar.close()

    print("\n5.3 文件处理进度条:")
    # 模拟文件处理
    files = [f"file_{i}.txt" for i in range(20)]
    for filename in tqdm(files, desc="处理文件"):
        # 模拟文件处理时间
        time.sleep(0.1)
        if filename.endswith("5.txt") or filename.endswith("15.txt"):
            tqdm.write(f"✓ 特殊处理完成: {filename}")


def main():
    """主函数"""
    print("🚀 tqdm进度条库完整演示")
    print("基于博客: https://blog.csdn.net/AI_dataloads/article/details/134169038")
    print("=" * 60)

    try:
        # 基础用法演示
        demo_basic_usage()

        # 描述设置演示
        demo_description()

        # write方法演示
        demo_write_method()

        # 深度学习应用演示
        demo_deep_learning()

        # 高级功能演示
        demo_advanced_features()

        print("\n" + "=" * 60)
        print("🎉 所有演示完成！")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")


if __name__ == "__main__":
    main()