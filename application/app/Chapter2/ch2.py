import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from application.app.Classes.AdalineGD import AdalineGD
from application.app.Classes.AdalineSGD import AdalineSGD
from application.app.Classes.Perceptron import Perceptron


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolor='black'
        )


# read in iris example data
file = os.path.join('..', '..', 'data', 'input', 'iris.data')
df = pd.read_csv(file, header=None, encoding='utf-8')
print(df.tail())

# select setosa and versicolor
y = df.iloc[:100, 4].values

# uncomment next line to classify between Iris-setosa and Iris-virginica
#y = df.iloc[list(range(50)) + list(range(100, 150)), 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[:100, [0, 2]].values

# uncomment next line to classify between Iris-setosa and Iris-virginica
#X = df.iloc[list(range(50)) + list(range(100, 150)), [0, 2]].values

# plot data
# adapt label in case of Iris-virginica
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:, 0], X[50:, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# train perceptron classifier
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

# plot errors
plt.figure()
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plot decision regions
plt.figure()
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# train AdalineGD classifier with eta = 0.01
ada1 = AdalineGD(n_iter=10, eta=0.01)
ada1.fit(X, y)

# plot errors
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

# train AdalineGD classifier with eta = 0.0001
ada2 = AdalineGD(n_iter=10, eta=0.0001)
ada2.fit(X, y)

# plot errors
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

# standardize data
X_std = np.copy(X)
# X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
# #X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# implementation for multiple columns, without loop
X_std = (X_std - X_std.mean(axis=0)) / X_std.std(axis=0)

# train AdalineGD classifier on standardized data
ada_gd = AdalineGD(n_iter=15, eta=0.01)
ada_gd.fit(X_std, y)

# plot error
ax[2].plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o')
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('Sum-squared-error')
ax[2].set_title('Adaline - Learning rate 0.01 - standardization')

# plot decision regions
plt.figure()
plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()

# train AdalineSGD classifier on standardized data
ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)

# plot decision regions
plt.figure()
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

# plot errors
plt.figure()
plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.title('Adaline - SGD - eta 0.01 - standardization')

plt.show()
