import numpy as np
from sklearn.metrics import confusion_matrix    # 混淆矩阵
from sklearn.metrics import f1_score            # f1 得分
from sklearn.metrics import cohen_kappa_score   # kappa 系数


def mean_iou(input, target, classes=1):
    """  compute the value of mean iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        miou: float, the value of miou
    """
    miou = 0
    for i in range(classes):
        intersection = np.logical_and(target == i, input == i)
        # print(intersection.any())
        union = np.logical_or(target == i, input == i)
        temp = np.sum(intersection) / np.sum(union)
        miou += temp
    return miou / classes


def compute_f1(prediction, target):
    """
    f1 score is the same as dice ceof in math
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    prediction.tolist(), target.tolist()
    img, target = np.array(prediction).flatten(), np.array(target).flatten()
    f1 = f1_score(y_true=target, y_pred=img)
    return f1


def compute_kappa(prediction, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        kappa: float
    """
    prediction.tolist(), target.tolist()
    img, target = np.array(prediction).flatten(), np.array(target).flatten()
    kappa = cohen_kappa_score(target, img)
    return  kappa

def compute_acc(gt, pred):
    '''
    compute accuracy
    :param gt: ground truth
    :param pred: prediction
    :return: accuracy
    '''
    matrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
    acc = np.diag(matrix).sum() / matrix.sum()
    return acc


if __name__ == "__main__":
    print()     # 占位