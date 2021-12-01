from matplotlib import pyplot as plt
import numpy as np
import math


def main():
    q2a(0.5, 0.25, 20, 0.5, 1.05, 0.05)
    q2b()


def q2a(ex, vx, n, a_start, a_stop, a_step):
    ex = 0.5
    vx = 0.25
    n = 20

    def emfreq():
        y_vals = (np.random.binomial(20, 0.5, 1000000) / 20)
        alphas = np.arange(a_start, a_stop, a_step)
        ps = np.zeros((len(alphas)))
        for n in range(0, len(alphas)):
            alpha = alphas[n]
            ps[n] = (y_vals >= alpha).sum() / 1000000
        return ps

    def markov(a): return ex / a
    def cheb(a): return vx / (n * ((a - ex) ** 2))
    def hoff(a): return 1 / (math.e ** (2 * n * ((a - ex) ** 2)))

    alpha = np.arange(a_start, a_stop, a_step)

    freq_y = emfreq()
    markov_y = np.array(list(map(markov, alpha)))
    cheb_y = np.array(list(map(cheb, alpha)))
    hoff_y = np.array(list(map(hoff, alpha)))

    plt.plot(alpha, freq_y, label="Empirical frequency")
    plt.plot(alpha, markov_y, label="Markov")
    plt.plot(alpha, cheb_y, label="Chebyshev")
    plt.plot(alpha, hoff_y, label="Hoeffding")
    plt.ylabel(r"$\mathbb{P}(\frac{1}{20}\sum^{20}_{i = 1}{X_i} \geq \alpha)$")
    plt.xlabel("\u03B1")
    plt.title("Bias of 0.5")
    plt.ylim(0, 1)
    plt.xlim(0.5, 1)
    plt.legend()
    plt.savefig("./Plots/plot_q3_1", dpi=200)
    plt.cla()


def q2b():
    ex = 0.1
    vx = 0.09
    n = 20

    def emfreq():
        y_vals = (np.random.binomial(20, 0.1, 1000000) / 20)
        alphas = np.arange(0.1, 1.05, 0.05)
        ps = np.zeros((19))
        for n in range(0, len(alphas)):
            alpha = alphas[n]
            ps[n] = (y_vals >= alpha).sum() / 1000000
        return ps

    def markov(a): return ex / a
    def cheb(a): return vx / (n * ((a - ex) ** 2))
    def hoff(a): return 1 / (math.e ** (2 * n * ((a - ex) ** 2)))

    alpha = np.arange(0.1, 1.05, 0.05)

    freq_y = emfreq()
    markov_y = np.array(list(map(markov, alpha)))
    cheb_y = np.array(list(map(cheb, alpha)))
    hoff_y = np.array(list(map(hoff, alpha)))

    plt.plot(alpha, freq_y, label="Empirical frequency")
    plt.plot(alpha, markov_y, label="Markov")
    plt.plot(alpha, cheb_y, label="Chebyshev")
    plt.plot(alpha, hoff_y, label="Hoeffding")
    plt.legend()
    plt.ylabel(r"$\mathbb{P}(\frac{1}{20}\sum^{20}_{i = 1}{X_i} \geq \alpha)$")
    plt.xlabel("\u03B1")
    plt.ylim(0, 1)
    plt.xlim(0.1, 1)
    plt.title("Bias of 0.1")
    plt.savefig("./Plots/plot_q3_2", dpi=200)


if __name__ == '__main__':
    main()

# Q7
# For alpha = 1, p = 1/1048576
# For alpha = 0.95, p = 5/262144
