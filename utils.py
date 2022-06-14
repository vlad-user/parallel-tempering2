import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage.filters import gaussian_filter1d
import tensorflow as tf
import wandb


def augment_images(inputs, istrain):

  def true_fn(x):
    with tf.device('cpu:0'):
      maybe_flipped = tf.image.random_flip_left_right(x)
      padded = tf.pad(maybe_flipped, [[0, 0], [4, 4], [4, 4], [0, 0]])
      cropped = tf.image.random_crop(padded, size=tf.shape(x))
    return cropped

  return tf.cond(istrain,
                 true_fn=lambda: true_fn(inputs),
                 false_fn=lambda: tf.identity(inputs))


def numpy_ewma_vectorized(data, window):
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha

    scale = 1 / alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale ** r
    offset = data[0] * alpha_rev ** (r + 1)
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


def plot_up_down_ratios(ratios):
    plt.plot(sorted(ratios.keys()), [ratios[k] for k in sorted(ratios.keys())], '-o')
    plt.xlabel('T value')
    plt.ylabel('f(T)')
    plt.show()


def apply_filter(x, y, sigma=1):
    ynew = gaussian_filter1d(y, sigma=sigma)
    return x, ynew


def plot_error(n_replicas, batch_size, train_data_size, noise_list, swap_history, noswap_history=None):
    test_swap_errors = [[(1 - val) * 100 for val in swap_history[f'val_acc_{i}']] for i in range(n_replicas)]
    train_swap_errors = [[(1 - val) * 100 for val in swap_history[f'acc_{i}']] for i in range(n_replicas)]

    if noswap_history:
        test_noswap_errors = [[(1 - val) * 100 for val in noswap_history[f'val_acc_{i}']] for i in range(n_replicas)]
        train_noswap_errors = [[(1 - val) * 100 for val in noswap_history[f'acc_{i}']] for i in range(n_replicas)]
    else:
        train_noswap_errors, test_noswap_errors = None, None

    def plot(n_replicas, batch_size, train_data_size, noise_list, swap_errors, noswap_errors):

        EPOCH_MULT = np.ceil(train_data_size/batch_size)
        LINEWIDTH = 5
        TLINEWIDTH = 3
        alpha = 0.35
        sigma = 4
        LEGEND_SIZE = 21
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        min_err_swap = np.min(np.array(swap_errors))
        swap_minrid = np.argmin(np.min(np.array(swap_errors), axis=1))

        if noswap_errors:
            min_err_noswap = np.min(np.array(noswap_errors))
            noswap_minrid = np.argmin(np.min(np.array(noswap_errors), axis=1))

        yswap = swap_errors[swap_minrid]
        xswap = np.arange(0, len(yswap))

        if noswap_errors:
            ynoswap = noswap_errors[noswap_minrid]
            xnoswap = np.arange(0, len(ynoswap))

        fig, ax = plt.subplots(figsize=(12, 8))


        yswap_orig = yswap.copy()
        if noswap_errors:
            ynoswap_orig = ynoswap.copy()

        xswap, yswap = apply_filter(xswap, yswap, sigma=sigma)
        if noswap_errors:
            xnoswap, ynoswap = apply_filter(xnoswap, ynoswap, sigma=sigma)

        label = '$\gamma\in {0} {1:.3f}, ..., {2:.3f} {3}^{4} $'.format(
            '\{', min(noise_list), max(noise_list), '\}', n_replicas)
        ax.plot(xswap, yswap, label=label, color=colors[0], linewidth=LINEWIDTH)
        ax.plot(xswap, yswap_orig, alpha=alpha, linewidth=TLINEWIDTH, color=colors[0])

        if noswap_errors:
            ax.plot(xnoswap, ynoswap, label='$\gamma^*={0:.3f}$)'.format(noise_list[noswap_minrid]),
                color=colors[1], linewidth=LINEWIDTH)
            ax.plot(xnoswap, ynoswap_orig, alpha=alpha, linewidth=TLINEWIDTH, color=colors[1])


        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        FONTSIZE = 25


        # top_line_y = [31, 31]
        # ax.plot([0, 404*EPOCH_MULT], top_line_y, linestyle='--', color='black', linewidth=2.5, dashes=(8, 12))
        # top_line_y = [29, 29]
        # ax.plot([0, 404*EPOCH_MULT], top_line_y, linestyle='--', color='black', linewidth=2.5, dashes=(8, 12))
        # top_line_y = [27, 27]
        # ax.plot([0, 404*EPOCH_MULT], top_line_y, linestyle='--', color='black', linewidth=2.5, dashes=(8, 12))

        # plt.yticks([27, 29, 31])
        # xticks = [20000, 40000, 60000, 80000, 100000, 120000, 140000]
        # xlabels = ['20K', '40K', '60K', '80K', '100K', '120K', '140K']

        # plt.ylim((26.95, 31.05))
        # plt.xlim((0, 404*EPOCH_MULT))


        plt.yticks(fontsize=23)
        # plt.xticks(xticks, xlabels, fontsize=23)
        plt.xlabel('Epochs', fontsize=FONTSIZE)
        plt.ylabel('Error (%)', fontsize=FONTSIZE)
        plt.rcParams["legend.loc"] = 'lower left'
        leg = plt.legend(fancybox=True, prop={'size': 19})
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linewidth(3)
        ax.set_rasterized(True)

        dirname = os.path.join(os.getcwd(), 'plots')

        if not os.path.exists(dirname):
          os.makedirs(dirname)

        path = os.path.join(dirname, 'cifar-learning_rate.eps')

        plt.savefig(path, bbox_inches='tight')

        return plt

    train_plt = plot(n_replicas, batch_size, train_data_size, noise_list, train_swap_errors, train_noswap_errors)
    test_plt = plot(n_replicas, batch_size, train_data_size, noise_list, test_swap_errors, test_noswap_errors)
    return train_plt, test_plt


def plot_error_2(n_replicas, histories, labels, noise_list):

    test_errors = [[[(1 - val) * 100 for val in h[f'val_acc_{i}']] for i in range(n_replicas)] for h in histories]
    train_errors = [[[(1 - val) * 100 for val in h[f'acc_{i}']] for i in range(n_replicas)] for h in histories]


    def plot(n_replicas, errors, labels, noise_list):

        LINEWIDTH = 5
        TLINEWIDTH = 3
        alpha = 0.35
        sigma = 4
        LEGEND_SIZE = 21
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i, e in enumerate(errors):

            min_err = np.min(np.array(e))
            minrid = np.argmin(np.min(np.array(e), axis=1))

            y = e[minrid]
            x = np.arange(0, len(y))


            fig, ax = plt.subplots(figsize=(12, 8))


            y_orig = y.copy()

            x, y = apply_filter(x, y, sigma=sigma)



            if labels[i] == 'no swap':
                label = '$\gamma^*={0:.3f}$)'.format(noise_list[minrid])
            else:
                label = '$\gamma\in {0} {1:.3f}, ..., {2:.3f} {3}^{4} $'.format(
                    '\{', min(noise_list), max(noise_list), '\}', n_replicas)

            ax.plot(x, y, label=label, color=colors[i], linewidth=LINEWIDTH)
            ax.plot(x, y_orig, alpha=alpha, linewidth=TLINEWIDTH, color=colors[i])


        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        FONTSIZE = 25


        # top_line_y = [31, 31]
        # ax.plot([0, 404*EPOCH_MULT], top_line_y, linestyle='--', color='black', linewidth=2.5, dashes=(8, 12))
        # top_line_y = [29, 29]
        # ax.plot([0, 404*EPOCH_MULT], top_line_y, linestyle='--', color='black', linewidth=2.5, dashes=(8, 12))
        # top_line_y = [27, 27]
        # ax.plot([0, 404*EPOCH_MULT], top_line_y, linestyle='--', color='black', linewidth=2.5, dashes=(8, 12))

        # plt.yticks([27, 29, 31])
        # xticks = [20000, 40000, 60000, 80000, 100000, 120000, 140000]
        # xlabels = ['20K', '40K', '60K', '80K', '100K', '120K', '140K']

        # plt.ylim((26.95, 31.05))
        # plt.xlim((0, 404*EPOCH_MULT))


        plt.yticks(fontsize=23)
        # plt.xticks(xticks, xlabels, fontsize=23)
        plt.xlabel('Epochs', fontsize=FONTSIZE)
        plt.ylabel('Error (%)', fontsize=FONTSIZE)
        plt.rcParams["legend.loc"] = 'lower left'
        leg = plt.legend(fancybox=True, prop={'size': 19})
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linewidth(3)
        ax.set_rasterized(True)

        # dirname = os.path.join(os.getcwd(), 'plots')
        #
        # if not os.path.exists(dirname):
        #   os.makedirs(dirname)
        #
        # path = os.path.join(dirname, 'cifar-learning_rate.eps')
        #
        # plt.savefig(path, bbox_inches='tight')

        return plt

    train_plt = plot(n_replicas, train_errors, labels, noise_list)
    test_plt = plot(n_replicas, test_errors, labels, noise_list)

    return train_plt, test_plt