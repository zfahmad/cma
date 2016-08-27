import os

import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import numpy as np
from matplotlib.patches import Ellipse

import cma


def rose(x, y):
    return 1 * (((1 - x) ** 2) + 100 * ((y - x ** 2) ** 2))


def sphere(x, y):
    return x ** 2 + y ** 2


mu = 10
lam = 30

dir = "./plots"

if not os.path.exists(dir):
    os.mkdir(dir)

options = {"bounds": [[-3, -3], [3, 3]], "mu": mu}
cm = cma.CMA(func=cma.rosenbrock, options=options)

C = np.eye(2, 2)
wi, he, an = cm.create_2d_ellipse()
cm.multi_gaussian(lam)
cm.choose_parents()

plt.tick_params(axis='x',
                which='both',
                bottom='off',
                top='off',
                labelbottom='off')

plt.tick_params(axis='y',
                which='both',
                left='off',
                right='off',
                labelleft='off')

# plt.xlabel(r'$\mathcal{X}$', size=20)
# plt.ylabel(r'$f:\mathcal{X}\rightarrow\mathbb{R}$', size=20)



# f = open(dir + "/cov_0.txt","w")
# g = open(dir + "/off_0.txt", "w")
# h = open(dir + "/par_0.txt", "w")
#
# f.write(str(cm.m) + " " + str(wi) + " " + str(he) + " " + str(an))
#
# for j in range(np.size(cm.offspring, axis=1)):
#     g.write(str(cm.offspring[0, j]) + " " + str(cm.offspring[1, j]) + "\r")
#
# for j in range(np.size(cm.parents, axis=1)):
#     h.write(str(cm.parents[0, j]) + " " + str(cm.parents[1, j]) + "\r")
#
# f.close()
# g.close()
# h.close()
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim(-5, 3)
ax.set_ylim(-4, 4)
ax.scatter(cm.m[0], cm.m[1], color='black', marker='+', lw=3, s=100)
ax.scatter(1, 1, color='red', marker='x', lw=3, s=100)
ell = Ellipse(xy=cm.m, width=wi, height=he, angle=an)
ax.add_artist(ell)
ell.set_fill(False)
# ell.set_alpha(0.5)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.axhline(0, color='black', ls="--", alpha=0.7)
ax.axvline(-1, color='black', ls="--", alpha=0.7)
plt.savefig(dir + "/cov_0")

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim(-5, 3)
ax.set_ylim(-4, 4)
ax.scatter(1, 1, color='red', marker='x', lw=3, s=100)
ax.scatter(cm.offspring[0, :], cm.offspring[1, :], color='blue', alpha=0.7, s=25)
ell = Ellipse(xy=cm.m, width=wi, height=he, angle=an)
ax.add_artist(ell)
ell.set_fill(False)
ell.set_alpha(0.5)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.axhline(0, color='black', ls="--", alpha=0.7)
ax.axvline(-1, color='black', ls="--", alpha=0.7)
plt.savefig(dir + "/off_0")

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim(-5, 3)
ax.set_ylim(-4, 4)
ax.scatter(1, 1, color='red', marker='x', lw=3, s=100)
ax.scatter(cm.parents[0, :], cm.parents[1, :], color='green', alpha=0.8, s=25)
ell = Ellipse(xy=cm.m, width=wi, height=he, angle=an)
ax.add_artist(ell)
ell.set_fill(False)
ell.set_alpha(0.5)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.axhline(0, color='black', ls="--", alpha=0.7)
ax.axvline(-1, color='black', ls="--", alpha=0.7)
plt.savefig(dir + "/par_0")
# plt.savefig(dir + "/cma_0")

for i in range(2):
    cm.update_mean()
    cm.update_cov()
    wi, he, an= cm.create_2d_ellipse()

    cm.multi_gaussian(lam)
    cm.choose_parents()

    # f = open(dir + "/cov_" + str(i+1) + ".txt", "w")
    # g = open(dir + "/off_" + str(i+1) + ".txt", "w")
    # h = open(dir + "/par_" + str(i+1) + ".txt", "w")
    #
    # f.write(str(cm.m) + " " + str(wi) + " " + str(he) + " " + str(an))
    #
    # for j in range(np.size(cm.offspring, axis=1)):
    #     g.write(str(cm.offspring[0, j]) + " " + str(cm.offspring[1, j]) + "\r")
    #
    # for j in range(np.size(cm.parents, axis=1)):
    #     h.write(str(cm.parents[0, j]) + " " + str(cm.parents[1, j]) + "\r")
    #
    # f.close()
    # g.close()
    # h.close()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim(-5, 3)
    ax.set_ylim(-4, 4)
    ax.scatter(cm.m[0], cm.m[1], color='black', marker='+', lw=3, s=100)
    ax.scatter(1, 1, color='red', marker='x', lw=3, s=100)
    ell = Ellipse(xy=cm.m, width=wi, height=he, angle=an)
    ax.add_artist(ell)
    ell.set_fill(False)
    # ell.set_alpha(0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axhline(0, color='black', ls="--", alpha=0.7)
    ax.axvline(-1, color='black', ls="--", alpha=0.7)
    plt.savefig(dir + "/cov_" + str(i+1))

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim(-5, 3)
    ax.set_ylim(-4, 4)
    ax.scatter(1, 1, color='red', marker='x', lw=3, s=100)
    ax.scatter(cm.offspring[0, :], cm.offspring[1, :], color='blue', alpha=0.7, s=25)
    ell = Ellipse(xy=cm.m, width=wi, height=he, angle=an)
    ax.add_artist(ell)
    ell.set_fill(False)
    ell.set_alpha(0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axhline(0, color='black', ls="--", alpha=0.7)
    ax.axvline(-1, color='black', ls="--", alpha=0.7)
    plt.savefig(dir + "/off_" + str(i+1))

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim(-5, 3)
    ax.set_ylim(-4, 4)
    ax.scatter(1, 1, color='red', marker='x', lw=3, s=100)
    ax.scatter(cm.parents[0, :], cm.parents[1, :], color='green', alpha=0.8, s=25)
    ell = Ellipse(xy=cm.m, width=wi, height=he, angle=an)
    ax.add_artist(ell)
    ell.set_fill(False)
    ell.set_alpha(0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axhline(0, color='black', ls="--", alpha=0.7)
    ax.axvline(-1, color='black', ls="--", alpha=0.7)
    plt.savefig(dir + "/par_" + str(i+1))

