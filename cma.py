import numpy as np
import scipy.linalg
import cholesky as ch
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math

def rosenbrock(x):
    return (((1 - x[0])**2) + 10 * ((x[1] - x[0]**2)**2))

class CMA:
    def __init__(self,  func, options=None,):
        self.params = {"bounds": [[0, 0], [1, 1]],
                       "max_iter": 500,
                       "lam": 2,
                       "mu": 1,
                       "c_m": 1,
                       "c_1": 0.5,
                       "c_c": 0.6,
                       "mu_eff": 5.5,
                       "sigma": 0.5}

        self.func = func
        self.offspring = np.array([])
        self.parents = np.array([])
        self.fitness = np.array([])
        self.m_ = np.array([])

        if options is not None:
            for k, v in options.items():
                self.params[k] = v

        self.x = None
        self.m = np.array([[0], [0]])
        self.p_c = np.array([[0], [0]])

    def multi_gaussian(self, m, C, num_samples=1):
        v, L = np.linalg.eig(C)
        u = np.random.normal(0, 1, (np.size(L, axis=0), 1))
        X = m + np.dot(L, u)
        Y = self.func(X)

        for i in range(num_samples - 1):
            u = np.random.normal(0, 1, (np.size(L, axis=0), 1))
            x = m + np.dot(L, u)
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
        self.parents = self.offspring[:, range(-1, -self.params["mu"] - 1, -1)]

    def update_mean(self):
        self.m_ = self.m
        self.m = self.m + (self.parents - self.m)

    def rank_one_update(self, C, m):
        # y = (x - self.m) / self.params['sigma']
        c_c = self.params['c_c']
        mu_eff = self.params["mu_eff"]
        sig = self.params["sigma"]

        self.p_c = (1 - c_c) + math.sqrt(c_c * (2 - c_c) * mu_eff) * (self.m - self.m_) / sig

        C = (1 - self.params['c_1']) * C + self.params['c_1'] * np.dot(self.p_c, self.p_c.T)

        return C

    def plot_2d(self, m, K):
        K = self.params["sigma"] * K
        x, y = np.mgrid[-1:1.01:.01, -1:1.01:.01]
        pos = np.dstack((x, y))
        rv = multivariate_normal(m, K)
        #a, b = np.random.multivariate_normal(m, K/15, 10).T
        fig2 = plt.figure(figsize=(7,7))
        ax2 = fig2.add_subplot(111)
        ax = fig2.gca()
        ax.set_xlim(-.5, .5)
        ax.set_ylim(-.15, 1)
        plt.arrow(0, 0, -.14, .41, length_includes_head=True, alpha=0.5)
        plt.arrow(-.14, .41, .23 + .14, .67 - .41, length_includes_head=True, alpha=0.5)
        plt.arrow(.23, .67, .1 - .23, .8 - .67, length_includes_head=True, alpha=0.5)
        plt.arrow(.1, .8, .13 - .1, .82 - .8, length_includes_head=True, alpha=0.5)
        plt.arrow(.13, .82, .19 - .13, .81 - .82, length_includes_head=True, alpha=0.5)
        ax2.contour(x, y, rv.pdf(pos), cmap=plt.cm.OrRd)
        #plt.scatter(a, b, marker="x", color="blue", alpha=0.3)
        plt.scatter(0.22, 0.85, marker="x", color="red", s=100)
        #plt.scatter(0, 0, marker="+", color="black", s=100)
        plt.axhline(0, color="black", alpha=0.8)
        plt.axvline(0, color="black", alpha=0.8)
        if not self.x is None:
            plt.scatter(self.x[0], self.x[1])
        plt.grid()
        plt.show()



options = {"bounds": [[1, 1], [3, 3]]}
cma = CMA(func=rosenbrock, options=options)

y = np.array([[1], [1]])
# C = np.dot(y, y.T)
C = np.eye(2, 2)
m = np.array([[0.5], [0.5]])
cma.multi_gaussian(m, C, 10)
cma.choose_parents()
#
# print(C)
# #cma.plot_2d(cma.m.T[0], C)
#
# cma.x = np.array([[-.14], [.41]])
# m = cma.m
# cma.update_mean(cma.x)
# print(cma.m)
# C = cma.rank_one_update(C, m)
# print(C)
# #cma.plot_2d(cma.m.T[0], C)
#
# cma.x = np.hstack((cma.x, np.array([[.23], [.67]])))
# m = cma.m
# cma.update_mean(np.array([[.23], [.67]]))
# print(cma.m)
# C = cma.rank_one_update(C, m)
# print(C)
# #cma.plot_2d(cma.m.T[0], C)
#
# cma.x = np.hstack((cma.x, np.array([[.1], [.8]])))
# m = cma.m
# cma.update_mean(np.array([[.1], [.8]]))
# print(cma.m)
# C = cma.rank_one_update(C, m)
# print(C)
# #cma.plot_2d(cma.m.T[0], C)
#
# cma.x = np.hstack((cma.x, np.array([[.13], [.82]])))
# m = cma.m
# cma.update_mean(np.array([[.13], [.82]]))
# print(cma.m)
# C = cma.rank_one_update(C, m)
# print(C)
# #cma.plot_2d(cma.m.T[0], C)
#
# cma.x = np.hstack((cma.x, np.array([[.19], [.81]])))
# m = cma.m
# cma.update_mean(np.array([[.19], [.81]]))
# print(cma.m)
# C = cma.rank_one_update(C, m)
# print(C)
# cma.plot_2d(cma.m.T[0], C)