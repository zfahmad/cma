import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math


def rosenbrock(x):
    return -(((1 - x[0]) ** 2) + 100 * ((x[1] - x[0] ** 2) ** 2))


def sphere(x):
    return -(x[0] ** 2 + x[1] ** 2)


class CMA:
    def __init__(self, func, options=None, ):
        self.params = {"bounds": np.array([[0, 0], [1, 1]]),
                       "max_iter": 500,
                       "lam": 2,
                       "mu": 2,
                       "c_m": 1,
                       "c_1": 0.6,
                       "c_c": 1,
                       "c_mu": 0.3,
                       "mu_eff": 5.5,
                       "sigma": 1}

        self.func = func
        self.offspring = np.array([])
        self.parents = np.array([])
        self.fitness = np.array([])
        self.C = np.eye(np.size(self.params["bounds"], axis=0))

        if options is not None:
            for k, v in options.items():
                self.params[k] = v

        self.x = None
        self.m = np.array([[-1], [0]])
        self.m_ = self.m
        self.p_c = self.m

        self.old_angle = 0

    def multi_gaussian(self, num_samples=1):
        v, L = np.linalg.eig(self.C)
        v = np.eye(np.size(L, axis=0)) * v
        u = np.random.normal(0, (np.size(L, axis=0), 1))
        X = self.m + self.params["sigma"] * np.dot(L, np.dot(v, u))
        Y = self.func(X)

        for i in range(num_samples - 1):
            u = np.random.normal(0, 1, (np.size(L, axis=0), 1))
            x = self.m + self.params["sigma"] * np.dot(L, np.dot(v, u))
            y = self.func(x)
            X = np.hstack((X, x))
            Y = np.hstack((Y, y))

        self.offspring = X
        self.fitness = Y

        ordered_fitness = np.argsort(self.fitness, axis=0)
        ordered_offspring = self.offspring[:, ordered_fitness]

        self.offspring = ordered_offspring
        self.fitness = self.fitness[ordered_fitness]

    def choose_parents(self):
        self.parents = self.offspring[:, range(-1, -1 - self.params["mu"], -1)]

    def update_mean(self):
        self.m_ = self.m
        self.m = self.m + (
        1 / self.params['mu'] * np.reshape(np.sum(self.parents, axis=1),
                                           [2, 1]) - self.m)

    def rank_mu_update(self):
        y = (self.parents - self.m) / self.params['sigma']
        w = 1 / self.params['mu']

        C = w * np.dot(y, y.T)

        return C

    def rank_one_update(self):
        # y = (x - self.m) / self.params['sigma']
        c_c = self.params['c_c']
        mu_eff = self.params["mu_eff"]
        sig = self.params["sigma"]

        self.p_c = (1 - c_c) + math.sqrt(c_c * (2 - c_c) * mu_eff) * (
            self.m - self.m_) / sig

        C = self.params['c_1'] * np.dot(
            self.p_c, self.p_c.T)

        return C

    def update_cov(self):
        self.C = (1 - self.params['c_1'] - self.params['c_mu']) * self.C + \
                 self.rank_one_update() + self.rank_mu_update()

    def create_2d_ellipse(self):
        U, V = np.linalg.eig(self.C)
        width_vec = V[0, :]
        height_vec = V[1, :]

        width = np.sqrt(np.sum(width_vec ** 2))
        height = np.sqrt(np.sum(height_vec ** 2))

        sin_ = np.abs(width_vec[1]) / np.abs(width)
        angle = (np.arcsin(sin_) * 360) / (2 * np.pi)

        if width_vec[1] >= 0:
            if width_vec[0] < 0:
                angle = -(90 + (90 - angle))
            else:
                angle = -angle
        else:
            if width_vec[0] < 0:
                angle = -(180 + angle)
            else:
                angle = angle

        width = width * 2 * np.sqrt(5.991 * U[0])
        height = height * 2 * np.sqrt(5.991 * U[1])

        return width, height, angle

    def plot_2d(self, m, K, first=0):
        x, y = np.mgrid[-3:3.01:.01, -3:3.01:.01]
        pos = np.dstack((x, y))
        m_ = np.reshape(m, (1, 2))
        rv = multivariate_normal(m_[0], K)
        fig2 = plt.figure(figsize=(7, 7))
        ax2 = fig2.add_subplot(111)
        ax = fig2.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        if first != 0:
            plt.arrow(self.m_[0, 0], self.m_[1, 0],
                      self.m[0, 0] - self.m_[0, 0],
                      self.m[1, 0] - self.m_[1, 0], length_includes_head=True,
                      lw=1,
                      head_width=0.1)
        ax2.contour(x, y, rv.pdf(pos), cmap=plt.cm.gray)
        plt.scatter(1, 1, marker="x", color="red", s=200, lw=2)
        plt.scatter(self.m_[0, 0], self.m_[1, 0], marker="+", color="black",
                    s=100, lw=2)
        plt.axhline(0, color="black", alpha=0.8, ls="--")
        plt.axvline(0, color="black", alpha=0.8, ls="--")
        plt.scatter(self.offspring[0, :], self.offspring[1, :], s=5)
        ax.text(3, -3, 'Iteration: ' + str(first), verticalalignment="bottom",
                horizontalalignment='right', fontsize=20, color='green')
        plt.show()
