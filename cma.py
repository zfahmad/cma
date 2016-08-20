import numpy as np
import scipy.linalg
import cholesky as ch
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math

def rosenbrock(x):
    return -(((1 - x[0])**2) + 100 * ((x[1] - x[0]**2)**2))

class CMA:
    def __init__(self,  func, options=None,):
        self.params = {"bounds": np.array([[0, 0], [1, 1]]),
                       "max_iter": 500,
                       "lam": 2,
                       "mu": 2,
                       "c_m": 1,
                       "c_1": 0.6,
                       "c_c": 0.6,
                       "c_mu": 0.2,
                       "mu_eff": 5.5,
                       "sigma": 0.3}

        self.func = func
        self.offspring = np.array([])
        self.parents = np.array([])
        self.fitness = np.array([])
        self.C = np.eye(np.size(self.params["bounds"], axis=0))

        if options is not None:
            for k, v in options.items():
                self.params[k] = v

        self.x = None
        self.m = np.array([[0], [0]])
        self.m_ = np.array([[0], [0]])
        self.p_c = np.array([[0], [0]])

    def multi_gaussian(self, num_samples=1):
        v, L = np.linalg.eig(self.C)
        u = np.random.normal(0, 1, (np.size(L, axis=0), 1))
        X = self.m + self.params["sigma"] * np.dot(L, u)
        Y = self.func(X)

        for i in range(num_samples - 1):
            u = np.random.normal(0, 1, (np.size(L, axis=0), 1))
            x = self.m + self.params["sigma"] + np.dot(L, u)
            y = self.func(x)
            X = np.hstack((X, x))
            Y = np.hstack((Y, y))

        self.offspring = X
        print(self.offspring)
        self.fitness = Y

        ordered_fitness = np.argsort(self.fitness, axis=0)
        ordered_offspring = self.offspring[:, ordered_fitness]

        self.offspring = ordered_offspring
        self.fitness = self.fitness[ordered_fitness]

    def choose_parents(self):
        self.parents = self.offspring[:, range(-1, -1 - self.params["mu"], -1)]

    def update_mean(self):
        self.m_ = self.m
        self.m = self.m + (1/self.params['mu'] * np.reshape(np.sum(self.parents, axis=1), [2, 1]) - self.m)

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

        self.p_c = (1 - c_c) + math.sqrt(c_c * (2 - c_c) * mu_eff) * (self.m - self.m_) / sig

        C = (1 - self.params['c_1']) * self.C + self.params['c_1'] * np.dot(self.p_c, self.p_c.T)

        return C

    def update_cov(self):
        self.C = (1 - self.params['c_1'] - self.params['c_mu']) * self.C + \
        self.rank_one_update() + self.rank_mu_update()

    def plot_2d(self, m, K, first=0):
        # K = self.params["sigma"] * K
        x, y = np.mgrid[-3:3.01:.01, -3:3.01:.01]
        pos = np.dstack((x, y))
        m_ = np.reshape(m, (1, 2))
        rv = multivariate_normal(m_[0], K)
        #a, b = np.random.multivariate_normal(m, K/15, 10).T
        fig2 = plt.figure(figsize=(7,7))
        ax2 = fig2.add_subplot(111)
        ax = fig2.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        #print(cma.m_[0,0], cma.m_[1,0], cma.m[0,0] - cma.m_[0,0], cma.m[1,0] - cma.m_[1,0])
        if first != 0:
            plt.arrow(cma.m_[0,0], cma.m_[1,0], cma.m[0,0] - cma.m_[0,0], cma.m[1,0] - cma.m_[1,0], length_includes_head=True, lw=3)
        # plt.arrow(-.14, .41, .23 + .14, .67 - .41, length_includes_head=True, alpha=0.5)
        # plt.arrow(.23, .67, .1 - .23, .8 - .67, length_includes_head=True, alpha=0.5)
        # plt.arrow(.1, .8, .13 - .1, .82 - .8, length_includes_head=True, alpha=0.5)
        # plt.arrow(.13, .82, .19 - .13, .81 - .82, length_includes_head=True, alpha=0.5)
        ax2.contour(x, y, rv.pdf(pos), cmap=plt.cm.OrRd)
        # #plt.scatter(a, b, marker="x", color="blue", alpha=0.3)
        plt.scatter(1, 1, marker="x", color="red", s=200, lw=2)
        plt.scatter(self.m_[0,0], self.m_[1,0], marker="+", color="black", s=100, lw=2)
        plt.axhline(0, color="black", alpha=0.8, ls="--")
        plt.axvline(0, color="black", alpha=0.8, ls="--")
        #if not self.offspring is None:
        plt.scatter(self.offspring[0, :], self.offspring[1, :], s=5)
        # plt.grid(True)
        plt.savefig("cma_" + str(first))



options = {"bounds": [[1, 1], [3, 3]], "mu": 6}
cma = CMA(func=rosenbrock, options=options)

C = np.eye(2, 2)
cma.multi_gaussian(20)
cma.plot_2d(cma.m, cma.C)

for i in range(3):
    cma.choose_parents()
    cma.update_mean()
    C = cma.update_cov()
    cma.multi_gaussian(20)
    cma.plot_2d(cma.m, cma.C, first=i + 1)
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