import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from collections import Counter

matplotlib.use("TkAgg")


# 显示数据
def show_digit(dataset, idx):
    # 防越界
    if idx < 0 or idx > len(dataset) - 1:
        return
    # 打印数据
    X = dataset.iloc[:, 1:]
    y = dataset.iloc[:, 0]
    print("图像集形状：", X.shape)
    print("类别占比：", Counter(y))      # 好的数据集类别占比是均衡的
    print("当前图像标签：", y[idx])

    # 显示指定照片
    image = X.iloc[idx].values
    image = image.reshape(28, 28)   # 转换为图像
    plt.axis("off")     # 不显示坐标轴
    plt.imshow(image, cmap="gray")
    plt.show()

if __name__ == "__main__":
    # Step 1: 读取数据
    dataset = pd.read_csv("手写数字识别.csv")
    # show_digit(dataset, 1)

    print()

    # Step 2: 数据预处理(归一化)
    X = dataset.iloc[:, 1:] / 255   # 归一化
    y = dataset.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=66)

    # Step 3: 模型训练
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(X_train, y_train)

    # Step 4: 模型评估
    acc = estimator.score(X_test, y_test)
    print("模型准确率：", acc)
    # 保存模型
    # joblib.dump(estimator, "model/knn-mnist.pth")
    # # 加载模型
    # estimator = joblib.load("model/knn-mnist.pth")

    # Step 5: 模型预测
    img = plt.imread("<file>")
    y_pred = estimator.predict(img.reshape(1, -1)/255)
    print("预测结果：", y_pred)
