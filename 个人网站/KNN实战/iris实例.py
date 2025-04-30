from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# step 1: 加载数据
dataset = load_iris()

# step 2: 数据集切分
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=0)

# step 3: 特征处理
transfer = StandardScaler()
transfer.fit(X_train)
transfer.transform(X_train)
transfer.transform(X_test)

# Step 4: 模型实例化
model = KNeighborsClassifier(n_neighbors=1)


# Step 5: 交叉验证 + 网格搜索
params_grid = {"n_neighbors": [4, 5, 7, 8]}     # 是3分类故不使用3或3的倍数
estimator = GridSearchCV(estimator=model, param_grid=params_grid, cv=4)
estimator.fit(X_train, y_train)

print(estimator.best_score_)
print(estimator.best_estimator_)
print(estimator.best_params_)
print(estimator.cv_results_)

model = estimator.best_estimator_
model.fit(X_train, y_train)

# step 6: 模型评估
y_hat = model.predict(X_test)
print("acc: ", accuracy_score(y_test, y_hat))

# Step 7: 模型预测
x = [[5.1, 3.5, 1.4, 0.2]]
y_pred = model.predict(x)
print(y_pred)