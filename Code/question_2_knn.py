import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from matplotlib import pyplot as plt


def main():
    task_1()
    task_2()


def task_1():
    model = np.loadtxt('Code/Data/MNIST-5-6-Subset.txt')
    labels = np.loadtxt('Code/Data/MNIST-5-6-Subset-Labels.txt')

    model = model.reshape(1877, 784)
    training = model[:100]
    t_labels = labels[:100]

    err_for_k = []

    for n in [10, 20, 40, 80]:
        val_index = validation_sets_index(n, 5, 100)
        for i in range(0, len(val_index)):
            v_index = val_index[i]
            k_diff_ts = np.zeros((50))
            for j in range(0, len(v_index)):
                j_index = val_index[i][j]
                ks = knn_pred(training, t_labels, model[j_index], 51)
                labz = np.tile(labels[j_index], (len(ks)))
                diff_k = np.abs(ks - labz)
                k_diff_ts = k_diff_ts + diff_k
            err = (k_diff_ts / n)
            err_for_k.append(err)
            plt.plot(np.arange(1, 51), err, label=f"i = {i + 1}")
        plt.xlabel("K")
        plt.ylabel("Error")
        plt.legend(loc="upper left")
        plt.ylim(0, 1)
        plt.xlim(1, 51)
        plt.title(f"Error vs. K for n={n}")
        plt.savefig(f"Plots/plot_q2_{n}", dpi=200)
        plt.cla()

    e_res = np.reshape(err_for_k, (4, 5, 50))
    var_for_k = np.zeros((4, 50))
    for i in range(0, len(e_res)):
        x = e_res[i]
        for j in range(0, len(x.T)):
            y = x.T[j]
            var_for_k[i][j] = np.var(y)

    ns = [10, 20, 40, 80]
    for x in range(0, len(var_for_k)):
        n = var_for_k[x]
        plt.plot(np.arange(1, 51), n, label=f"n={ns[x]}")
    plt.title(f"Variance vs. K")
    plt.legend()
    plt.xlim(1, 51)
    plt.savefig(f"Plots/plot_q2_p3", dpi=200)
    plt.cla()


def task_2():
    model_c_light = np.loadtxt(
        'Code/Data/MNIST-5-6-Subset-Light-Corruption.txt')
    model_c_moderate = np.loadtxt(
        'Code/Data/MNIST-5-6-Subset-Moderate-Corruption.txt')
    model_c_heavy = np.loadtxt(
        'Code/Data/MNIST-5-6-Subset-Heavy-Corruption.txt')

    labels = np.loadtxt('Code/Data/MNIST-5-6-Subset-Labels.txt')

    pl_w_corruption(model_c_light, labels, "light", 80)
    pl_w_corruption(model_c_moderate, labels, "moderate", 80)
    pl_w_corruption(model_c_heavy, labels, "heavy", 80)


def pl_w_corruption(model, labels, c_level, n):
    model = model.reshape(1877, 784)
    training = model[:100]
    t_labels = labels[:100]

    val_index = validation_sets_index(n, 5, 100)
    for i in range(0, len(val_index)):
        v_index = val_index[i]
        k_diff_ts = np.zeros((50))
        for j in range(0, len(v_index)):
            j_index = val_index[i][j]
            ks = knn_pred(training, t_labels, model[j_index], 51)
            labz = np.tile(labels[j_index], (len(ks)))
            diff_k = np.abs(ks - labz)
            k_diff_ts = k_diff_ts + diff_k
        err = (k_diff_ts / n)
        plt.plot(np.arange(1, 51), err, label=f"i = {i + 1}")
    plt.xlabel("K")
    plt.ylabel("Error")
    plt.legend(loc="upper left")
    plt.ylim(0, 1)
    plt.xlim(1, 51)
    c_title = "" if c_level == None else f" {c_level} corruption, "
    c_file_title = "" if c_level == None else f"_corruption_{c_level}"
    if ():
        plt.title(f"Error vs. K for {c_title}n=80")
        plt.savefig(f"Plots/plot_q2_{n}n{c_file_title}", dpi=200)
    plt.cla()


def sort_by_x(training, labels, x):
    diffs = np.linalg.norm(training - np.tile(x, (len(training), 1)), axis=1)
    idxs = np.argsort(diffs)
    return np.take(labels, idxs)


def knn_pred(t_set, t_labels, x, max_k):
    def more(e):
        unique, counts = np.unique(labs[:e], return_counts=True)
        occ = dict(zip(unique, counts))
        return max(occ, key=occ.get)
    labs = sort_by_x(t_set, t_labels, x)
    ks = np.arange(1, max_k)
    o = np.array(list(map(more, ks)))
    return o


def validation_sets(s, n, i, offset):
    x = s.take(indices=range(offset, offset + (n * (i + 1))), axis=0)
    return np.reshape(x, (i, x[0].shape))


def validation_sets_index(n, i, offset):
    s = np.arange(offset + n, offset + (n * (i + 1)))
    return np.reshape(s, (i, n))


# Archived inefficient functions
# def knn(S, x, k):
#     s = sorted_distance(x, S)
#     unique, counts = np.unique(s[:, 0][:k], return_counts=True)
#     occ = dict(zip(unique, counts))
#     return max(occ, key=occ.get)


# def sorted_distance(x, X):
#     def k(elem):
#         return distance(x, elem[1])
#     return X[np.apply_along_axis(k, 1, X).argsort()]


# def distance(x1, x2): return np.linalg.norm(x1 - x2)

if __name__ == '__main__':
    main()
