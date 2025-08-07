from util.evaluate import *

import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.optim
import torch.nn as nn

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_val(model, train_loader, val_loader, device, epochs, optimizer, loss, save_path,
              save_current_model=False, visualize=False, save_evaluation=False):
    """
    train model and testify model
    :param model: pytorch model
    :param train_loader: train loader
    :param val_loader: val loader or test loader
    :param device: 'cpu' or 'cuda', string
    :param epochs:
    :param optimizer:
    :param loss:
    :param save_path: best model save path
    :param save_current_model: save current model, saved model path is save_path
    :param visualize: True or False, bool
    :param save_evaluation: True or False, bool
    :return: None
    """
    model = model.to(device)

    # 用于画图的 train 和 val
    plt_train_loss = []
    plt_val_loss = []

    plt_train_acc = []
    plt_val_acc = []

    plt_train_iou = []
    plt_val_iou = []

    plt_train_f1_score = []
    plt_val_f1_score = []

    # 记录最小的val loss
    min_val_loss = 65535


    # 开始训练
    for epoch in range(epochs):
        # 评估参数
        train_loss = 0.0
        val_loss = 0.0
        train_accuracy = 0.0
        val_accuracy = 0.0
        train_mean_iou = 0.0
        val_mean_iou = 0.0
        train_f1_score = 0.0
        val_f1_score = 0.0

        # 记录时间
        start_time = time.time()
        model.train()  # 模型调整为训练模式
        for x_batch, y_batch in train_loader:
            # 将数据存储到GPU上
            x, y = x_batch.to(device), y_batch.to(device)
            y_pred = model(x)

            # 模型衡量记录
            y_pred_numpy = y_pred.cpu().detach().numpy()
            y_pred_bool = y_pred_numpy > 0.5
            y_numpy = y.cpu().detach().numpy()
            train_batch_loss = loss(y_pred, y)
            train_batch_acc = compute_acc(y_numpy, y_pred_bool)
            train_batch_iou = mean_iou(y_numpy, y_pred_bool)
            train_batch_f1 = compute_f1(y_pred_bool, y_numpy)

            # 反向传播
            train_batch_loss.backward()
            optimizer.step()  # 更新模型
            optimizer.zero_grad()

            # 模型评价
            train_loss += train_batch_loss.cpu().item()
            train_accuracy += train_batch_acc
            train_mean_iou += train_batch_iou
            train_f1_score += train_batch_f1

        plt_train_loss.append(train_loss / train_loader.__len__())  # 记录每轮批次计算的平均值
        plt_train_acc.append(train_accuracy / train_loader.__len__())
        plt_train_iou.append(train_mean_iou / train_loader.__len__())
        plt_train_f1_score.append(train_f1_score / train_loader.__len__())

        # 验证模型
        model.eval()  # 调整为验证模式
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x, y = x_batch.to(device), y_batch.to(device)
                y_pred = model(x)

                # 模型衡量记录
                y_pred_numpy = y_pred.cpu().detach().numpy()
                y_pred_bool = y_pred_numpy > 0.5
                y_numpy = y.cpu().detach().numpy()
                val_batch_loss = loss(y_pred, y)
                val_batch_acc = compute_acc(y_numpy, y_pred_bool)
                val_batch_iou = mean_iou(y_pred_bool, y_numpy)
                val_batch_f1 = compute_f1(y_pred_bool, y_numpy)

                # 模型评价
                val_loss += val_batch_loss.cpu().item()
                val_accuracy += val_batch_acc
                val_mean_iou += val_batch_iou
                val_f1_score += val_batch_f1
        plt_val_loss.append(val_loss / val_loader.__len__())
        plt_val_acc.append(val_accuracy / test_loader.__len__())
        plt_val_iou.append(val_mean_iou / test_loader.__len__())
        plt_val_f1_score.append(val_f1_score / test_loader.__len__())

        if save_current_model:
           torch.save(model, os.path.join(save_path, f"Current_Model.pth"))

        # 保存模型
        if min_val_loss > val_loss:
            # 保存模型的定义
            torch.save(model, os.path.join(save_path, f"Best_Model.pth"))

        if save_evaluation:
            import json
            save = {
                "current epoch": f"{epoch}/{epochs}",
                "train loss": plt_train_loss,
                "test loss": plt_val_loss,
                "train accuracy": plt_train_acc,
                "test accuracy": plt_val_acc,
                "train iou": plt_train_iou,
                "test iou": plt_val_iou,
                "train f1 score": plt_train_f1_score,
                "test f1 score": plt_val_f1_score,
            }
            with open(os.path.join(save_path, "evaluation.json"), 'w') as f:
                json.dump(save, f)

        print("-" * 120)
        print(f"{'Epoch':<12} {'Time(s)':>10} {'Train Loss':>12} {'Val Loss':>12} {'Train Acc':>12} {'Val Acc':>12} {'Train IoU':>12} {'Val IoU':>12} {'Train F1':>12} {'Val F1':>12}")
        print("-" * 120)
        print(f"[{epoch:3}/{epochs:3}]  {(time.time() - start_time):10.2f}  "
              f"{plt_train_loss[-1]:12.6f} {plt_val_loss[-1]:12.6f}  "
              f"{plt_train_acc[-1]:12.6f} {plt_val_acc[-1]:12.6f}  "
              f"{plt_train_iou[-1]:12.6f} {plt_val_iou[-1]:12.6f}  "
              f"{plt_train_f1_score[-1]:12.6f} {plt_val_f1_score[-1]:12.6f}")
        print("-" * 120)


def load_model(model_path: str) -> nn.Module:
    '''
    resume to train model
    :param model_path: model path(model file name *.pth)
    :param epoch: epoch when the model training stop
    :param epochs: epoch sum
    :return: nn.Module
    '''
    return torch.load(model_path)

# 修改超参数，只修改以下内容，不要修改之上的内容
if __name__ == "__main__":
    from model.test import TestModel
    from dataset.dataset import load_ISIC2018

    seed_everything(22)

    train_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset, test_dataset = load_ISIC2018("/data/公共数据集/ISIC2018", train_transforms, test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    save_path = "/home/zlh/save/model"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 0.0001
    epochs = 300
    model = TestModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)
    # 学习率退火 调度器   (替换优化器，目前不支持)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15,
    #                                                        threshold=0.0001, min_lr=0)  # 固定步长衰减
    loss = nn.MSELoss()
    train_val(model, train_loader, test_loader, device, epochs, optimizer, loss, save_path, save_current_model=True, save_evaluation=True)
