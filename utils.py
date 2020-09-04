import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage.filters import gaussian_filter1d
import albumentations as A


def augment_image(img):
    transform = A.Compose([A.HorizontalFlip(),
                           A.PadIfNeeded(min_height=img.shape[0] + 8, min_width=img.shape[1] + 8),
                           A.RandomCrop(height=img.shape[0], width=img.shape[1])
                           ]
                          )
    tr_img = transform(image=img)['image']
    return tr_img


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

        if noswap_history:
            min_err_noswap = np.min(np.array(noswap_errors))
            noswap_minrid = np.argmin(np.min(np.array(noswap_errors), axis=1))

        yswap = swap_errors[swap_minrid]
        xswap = np.arange(0, len(yswap))

        if noswap_history:
            ynoswap = noswap_errors[noswap_minrid]
            xnoswap = np.arange(0, len(ynoswap))

        fig, ax = plt.subplots(figsize=(12, 8))


        yswap_orig = yswap.copy()
        if noswap_history:
            ynoswap_orig = ynoswap.copy()

        xswap, yswap = apply_filter(xswap, yswap, sigma=sigma)
        if noswap_history:
            xnoswap, ynoswap = apply_filter(xnoswap, ynoswap, sigma=sigma)

        label = '$\gamma\in {0} {1:.3f}, ..., {2:.3f} {3}^{4} $'.format(
            '\{', min(noise_list), max(noise_list), '\}', n_replicas)
        ax.plot(xswap, yswap, label=label, color=colors[0], linewidth=LINEWIDTH)
        ax.plot(xswap, yswap_orig, alpha=alpha, linewidth=TLINEWIDTH, color=colors[0])

        if noswap_history:
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