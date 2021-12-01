import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def main():
    X = np.array([np.arange(0, 6)]).T
    Y = np.array([0, 14, 21, 25, 35, 32]).T

    X = np.concatenate((X ** 2, X), axis=1)

    wh = fit_Wh(X, Y)

    x_fit = np.linspace(0, 10, 100)
    y_fit = (wh[0][0] * x_fit**2) + (wh[0][1] * x_fit)

    roots = np.roots(np.array([wh[0][0], wh[0][1], 0]))

    plt.plot(x_fit, y_fit, 'k')
    plt.scatter(
        np.arange(0, 6),
        np.array([0, 14, 21, 25, 35, 32]),
        marker="o"
    )
    plt.vlines(
        roots[0],
        ymin=00, ymax=32,
        label=f"returns at: {round(roots[0], 2)}m",
        linestyles="dashed"
    )
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()
    plt.title("Cannonball displacement")
    plt.xlabel("Distance from cannon (m)")
    plt.ylabel("Height (m)")
    plt.savefig("Plots/plot_q5", dpi=200)


def fit_Wh(X, Y):
    return np.linalg.inv(np.array([X.T @ X])) @ X.T @ Y


def ml_method():
    def fit_func(x, a, b):
        return a * (x ** 2) + b * x

    x = np.arange(0, 6)
    y = np.array([0, 14, 21, 25, 35, 32])
    params = curve_fit(fit_func, x, y)
    [a, b] = params[0]
    x_fit = np.linspace(x[0], 10, 100)
    y_fit = a * x_fit**2 + b * x_fit

    plt.plot(x, y, '.r')         # Data
    plt.plot(x_fit, y_fit, 'k')  # Fitted curve
    plt.savefig("Plots/plot_q5", dpi=200)


if __name__ == '__main__':
    main()
