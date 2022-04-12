import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def true_f(x):
    '''Asume the underlying true function f that dictates the
    relationship between x and y is:'''
    return 0.5 * x + np.sqrt(np.max(x, 0)) - np.cos(x) + 2

def f_hat(x, w):
    '''As a reminder, in polynomial regression we try to fit the
     following non-linear relationship between x and y.'''
    d = len(w) - 1
    # f'(x) = w_0 + w_1*x + w_2*x^2 + ... + w_d*x^d
    return np.sum(w * np.power(x, np.expand_dims(np.arange(d, -1, -1), 1)).T, 1)

if __name__ == '__main__':
    N = 1000

    # Pair Samples (x,y)
    x_max = 3
    x = x_max * (2 * np.random.rand(N) - 1)
    epsilon = np.random.randn(N) #Irreductible error (Noise)
    y = true_f(x) + epsilon

    # unseen test point (x_test,y_test)
    x_test = 3.2
    y_test = true_f(x_test) + np.random.randn()


    plt.figure(figsize=(12, 6))

    # Plot the (x, y) pair samples
    plt.scatter(x, y, s=10)
    # unseen test point
    plt.scatter(x_test, y_test, c='r', s=12)
    #underlying true function f(x) line
    x_range = np.linspace(-x_max, x_max, 1000) # x-axis range
    plt.plot(x_range, true_f(x_range), 'r', linewidth=1.5)

    plt.xlabel('x', size=12)
    plt.ylabel('y', size=12)
    plt.title("Original Problem")
    plt.grid()
    plt.legend(['True_f', 'samples', 'unseen (test) point'], fontsize=12)
    plt.xticks(np.arange(-x_max, x_max + 1))
    plt.tight_layout()

    '''
    We will model the problem with polynomial regressions of varying degrees of complexity. Let’s assume we could only 
    use n points (out of the 1,000) to train our polynomial regression model and we consider 
    four different regression models, one with degree d=1 (simple line), one with d=2, d=4, d=9.
    '''

    # use n points to train our model
    n = int(0.2 * N) #.02 * N
    print("Points used to train our model: ", n)

    # different regression models
    deg_complex = [1, 2, 4, 9]

    N_exp = 1
    fig, axs = plt.subplots(2, 3, sharey='all', figsize=(15, 9))
    colors = np.array(['tab:green', 'tab:purple', 'tab:cyan', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive'])
    label_leg = [r'$f$']
    for d in deg_complex:
        label_leg.append(r"$\hat{f}$ (d = " + str(d) + ")")

    '''
    If we randomly sample n points from the underlying population (different realizations of the training data) and 
    we repeat this experiment 6 times, this is a possible outcome we get.
    '''
    for row in range(2):
        for col in range(3):
            # Select samples of training data points for a specific realization (experiment)
            idx = np.random.permutation(N)[:n]
            x_train, y_train = x[idx], y[idx]

            # set the different regression models
            w = []
            for d in deg_complex:
                w.append(np.polyfit(x_train, y_train, d)) #list of lists of weights

            # n training data points
            axs[row, col].scatter(x_train, y_train, s=10, c='tab:gray')

            # underlying (unknown to us) true function f
            axs[row, col].plot(x_range, true_f(x_range), 'r', linewidth=1.5)

            # The fitting of the different models to different realizations of training data.
            # the prediction f'(x) of test (unseen) point x under each model
            for k in range(len(w)):
                axs[row, col].plot(x_range, f_hat(x_range, w[k]), colors[k], linewidth=1.5)

            # unseen (test) point
            axs[row, col].scatter(x_test, y_test, c='r', s=12)
            for k in range(len(w)):
                axs[row, col].scatter(x_test, f_hat(x_test, w[k]), c=colors[k], s=12)

            axs[row, col].set_xlabel('x', size=12)
            axs[row, col].set_ylabel('y', size=12)
            axs[row, col].grid()
            axs[row, col].legend(label_leg, fontsize=12)
            #axs[i, j].legend([r'$f$', r'$\hat{f}$ (d = 1)', r'$\hat{f}$ (d = 2)', r'$\hat{f}$ (d = 3)', r'$\hat{f}$ (d = 5)'], fontsize=12)
            axs[row, col].title.set_text('Experiment {}'.format(N_exp))
            N_exp += 1

    plt.show()