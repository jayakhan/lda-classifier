import numpy as np
import pandas as pd  # this library is used only in plotting
from datagenerator import generate_data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


class lda:
    """Fisher's linear discriminant classifier."""

    def __init__(self):
        """Initialize clusters."""
        self.X = []
        self.y = []
        self.y_hat = []
        self.mu_class0, self.mu_class1 = (
            np.empty((2, 2)),
            np.empty((2, 2)),
        )
        self.SW_one, self.SW_two, self.SW = (
            np.zeros((2, 2, 2)),
            np.zeros((2, 2, 2)),
            np.zeros((2, 2, 2)),
        )
        self.W = np.zeros((2, 2, 1))
        self.W0 = np.zeros((2))

    def fit(self, X):
        """Classify data based on maximum eigen value distance."""
        self.y = X[:, 2].reshape(-1, 1)
        self.X = X[:, 0:2]
        class_indexes = []
        for i in range(2):
            class_indexes.append(np.argwhere(self.y == i))

        # Calculating SW, W & W0 #
        self.mu_class0 = np.mean(self.X[class_indexes[0]], axis=0)
        self.mu_class1 = np.mean(self.X[class_indexes[1]], axis=0)

        for i in range(2):
            between_class1 = np.subtract(
                self.X[class_indexes[i]].reshape(-1, 2), self.mu_class0[i]
            )
            self.SW_one[i] = between_class1.T.dot(between_class1)
            between_class2 = np.subtract(
                self.X[class_indexes[i]].reshape(-1, 2), self.mu_class1[i]
            )
            self.SW_two[i] = between_class2.T.dot(between_class2)
            self.SW[i] = self.SW_one[i] + self.SW_two[i]
            self.W[i] = np.dot(
                np.linalg.pinv(self.SW[i]),
                np.subtract(self.mu_class1[i], self.mu_class0[i]).reshape(-1, 1),
            )
            self.W0[i] = -0.5 * np.dot(
                self.W[i].T, (self.mu_class0[i] + self.mu_class1[i])
            )

    def predict(self, X) -> np.ndarray:
        """Predict Labels."""
        predict = np.zeros((len(X)), dtype=int)
        Y = np.zeros((len(X), 2))

        for j in range(len(X)):
            for i in range(2):
                Y[j, i] = np.dot(self.W[i].T, X[j]) + self.W0[i]
            predict[j] = np.argmin(Y[j])
        return predict

    def plot_decision_boundary(self):
        """Plot decision boundary."""
        h = 0.02
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        Y = pd.DataFrame(self.y)

        # Define plot parameters
        class_0_patch = mpatches.Patch(color="#FFBF00", label="Class 0")
        class_1_patch = mpatches.Patch(color="#c1e7fc", label="Class 1")
        colors = {0: "#FFBF00", 1: "#c1e7fc"}
        cmap_light = ListedColormap(["#c1e7fc", "#f4c579"])

        # Visualize data
        _ = plt.scatter(
            self.X[:, 0],
            self.X[:, 1],
            c=Y[0].map(colors),
            zorder=3,
            linewidths=0.7,
            edgecolor="k",
        )
        plt.contour(xx, yy, Z, cmap=cmap_light)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.legend(handles=[class_0_patch, class_1_patch])
        plt.title(
            "LDA Decision Boundary", fontdict={"fontsize": 16},
        )
        plt.xlabel("X0", fontdict={"fontsize": 15})
        plt.ylabel("X1", fontdict={"fontsize": 15})
        plt.show()


# # Code to test
# if __name__ == "__main__":
#     samples = generate_data()
#     lda_obj = lda()
#     lda_obj.fit(samples)
#     y_hat = lda_obj.predict(lda_obj.X)
#     lda_obj.plot_decision_boundary()

