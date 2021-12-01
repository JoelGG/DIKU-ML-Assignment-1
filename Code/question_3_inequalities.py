from matplotlib import pyplot as plt
import numpy as np
import math


def main():
    q(0.5, 20, 0.5, 1.05, 0.05)  # bias of 0.5, 20 repetitions of coin flip
    q(0.1, 20, 0.1, 1.05, 0.05)  # bias of 0.1, 20 repetitions of coin flip


def q(ex, n, a_start, a_stop, a_step, reps=1000000):
    vx = ex * (1 - ex)

    def markov(a): return ex / a
    def cheb(a): return vx / (n * ((a - ex) ** 2))
    def hoff(a): return 1 / (math.e ** (2 * n * ((a - ex) ** 2)))

    def emfreq():
        y_vals = (np.random.binomial(n, ex, reps) / n)
        alphas = np.arange(a_start, a_stop, a_step)
        ps = np.zeros((len(alphas)))
        for i in range(0, len(alphas)):
            alpha = alphas[i]
            ps[i] = (y_vals >= alpha).sum() / reps
        return ps

    alpha = np.arange(a_start, a_stop, a_step)

    markov_y = np.array(list(map(markov, alpha)))
    cheb_y = np.array(list(map(cheb, alpha)))
    hoff_y = np.array(list(map(hoff, alpha)))
    freq_y = emfreq()

    plt.plot(alpha, freq_y, label="Empirical frequency")
    plt.plot(alpha, markov_y, label="Markov")
    plt.plot(alpha, cheb_y, label="Chebyshev")
    plt.plot(alpha, hoff_y, label="Hoeffding")
    plt.ylabel(r"$\mathbb{P}(\frac{1}{20}\sum^{20}_{i = 1}{X_i} \geq \alpha)$")
    plt.xlabel("\u03B1")
    plt.title(f"Bias of {ex}")
    plt.ylim(0, 1)
    plt.xlim(a_start, a_stop - a_step)
    plt.legend()
    plt.savefig(str(f"./Plots/plot_q3_{int(20 * ex)}.png"), dpi=200)
    plt.cla()


if __name__ == '__main__':
    main()

# Q7
# For alpha = 1, p = 1/1048576
# For alpha = 0.95, p = 5/262144
