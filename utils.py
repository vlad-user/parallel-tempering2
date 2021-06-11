import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage.filters import gaussian_filter1d
# import albumentations as A
import tensorflow as tf
import wandb
import functools
from keras.datasets import cifar10
from keras.utils import np_utils
from read_datasets import get_emnist_letters
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as shuffle_dataset


def prepare_data(args):
    if args.dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    elif args.dataset_name == 'emnist':
        x_train, y_train, x_test, y_test = get_emnist_letters()

        x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)),
           mode='constant', constant_values=0)
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)),
                         mode='constant', constant_values=0)
        y_train = np.int32(y_train) - 1
        y_test = np.int32(y_test) - 1


    x_train = np.float32(x_train) / 255.
    x_test = np.float32(x_test) / 255.


    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    x_train, y_train = shuffle_dataset(x_train, y_train, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    return x_train, y_train, x_val, y_val, x_test, y_test,


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

def log_additional_losses(history, config, swap):
    for i, step in enumerate(history['step']):
        for j in range(config.n_replicas):
            wandb.log({f'exchange_data_loss_{j}': history[f'loss_{j}'][i], 'batch': step} )
        wandb.log({'replica_order': history['replica_order'][i], 'batch': step})
        try:
            wandb.log({'num_misordered_temp': history['num_misordered_temp'][i], 'batch': step} )
        except KeyError:
            continue



def log_exchange_data_mh(history, config, swap):
    for i, step in enumerate(history['step']):
        if swap:
            wandb.log({'exchange probas': history['proba'][i], 'batch': step} )
            wandb.log({'acceptance ratio': history['accept_ratio'][i], 'batch': step} )
            wandb.log({'delta': history['delta'][i], 'batch': step} )
            wandb.log({'exchange_pair': history['exchange_pair'][i], 'batch': step} )
            wandb.log({'swaped': history['swaped'][i], 'batch': step} )
        for j in range(config.n_replicas):
            wandb.log({f'replica_{j}_{config.hp_to_swap}': history[j][config.hp_to_swap][i], 'batch': step} )
            wandb.log({f'exchange_loss_{j}': history[f'loss_{j}'][i], 'batch': step} )

    if swap:
        wandb.log({'num of exchange attempts': len(history['proba'])})
        wandb.log({'num of exchanges': history['swaped'].count(1)})


def log_exchange_data_mh_log_all_probas(history, config, swap):
    for i, step in enumerate(history['step']):
        if swap:
            wandb.log({'exchange probas': history['proba'][i], 'batch': step} )
            # wandb.log({'exchange probas for all t': history['all_probas'][i], 'batch': step} )
            wandb.log({'exchange deltas for all t': history['all_deltas'][i], 'batch': step} )

            wandb.log({'acceptance ratio': history['accept_ratio'][i], 'batch': step} )
            wandb.log({'delta': history['delta'][i], 'batch': step} )
            wandb.log({'exchange_pair': history['exchange_pair'][i], 'batch': step} )
            wandb.log({'swaped': history['swaped'][i], 'batch': step} )

        for j in range(config.n_replicas):
            wandb.log({f'replica_{j}_{config.hp_to_swap}': history[j][config.hp_to_swap][i], 'batch': step} )
            wandb.log({f'exchange_loss_{j}': history[f'loss_{j}'][i], 'batch': step} )

        wandb.log({'num_misordered_temp': history['num_misordered_temp'][i], 'batch': step})
        try:
            wandb.log({f'num_rear': history[f'num_rear'][i], 'batch': step})
        except KeyError:
            continue





    if swap:
        wandb.log({'num of exchange attempts': len(history['proba'])})
        wandb.log({'num of exchanges': history['swaped'].count(1)})

def log_exchange_data_mh_temp_sort_log_all_probas(history, config, swap):
    for i, step in enumerate(history['step']):
        if swap:
            wandb.log({'exchange probas': history['proba'][i], 'batch': step} )
            # wandb.log({'exchange probas for all t': history['all_probas'][i], 'batch': step} )
            wandb.log({'exchange deltas for all t': history['all_deltas'][i], 'batch': step} )

            wandb.log({'acceptance ratio': history['accept_ratio'][i], 'batch': step} )
            wandb.log({'delta': history['delta'][i], 'batch': step} )
            wandb.log({'exchange_pair': history['exchange_pair'][i], 'batch': step} )
            wandb.log({'swaped': history['swaped'][i], 'batch': step} )
            wandb.log({'temp_order': history['temp_order'][i], 'batch': step} )
            wandb.log({'num_misordered_temp': history['num_misordered_temp'][i], 'batch': step} )


        for j in range(config.n_replicas):
            wandb.log({f'replica_{j}_{config.hp_to_swap}': history[j][config.hp_to_swap][i], 'batch': step} )
            wandb.log({f'exchange_loss_{j}': history[f'loss_{j}'][i], 'batch': step} )

    if swap:
        wandb.log({'num of exchange attempts': len(history['proba'])})
        wandb.log({'num of exchanges': history['swaped'].count(1)})


def log_exchange_data_mh_temp_adj(history, config, swap):
    for i, step in enumerate(history['step']):
        if swap:
            wandb.log({'exchange probas': history['proba'][i], 'batch': step} )
            wandb.log({'acceptance ratio': history['accept_ratio'][i], 'batch': step} )
            wandb.log({'delta': history['delta'][i], 'batch': step} )
            wandb.log({'exchange_pair': history['exchange_pair'][i], 'batch': step} )
            wandb.log({'swaped': history['swaped'][i], 'batch': step} )
            # if i % config.temp_adj_step == 0 and step != 0:  #log history of proposed hp values
            #     for k in history:
            #         if str(k).startswith('hp_v_'):
            #             wandb.log({f'adj_value_for_{k}': history[k][(i // config.temp_adj_step) - 1], 'batch': step})
            if i % config.temp_adj_step == 0:  # log history of proposed hp values
                for k in history:
                    if step != 0:
                        if str(k).startswith('difference_beta_indx_'):
                            wandb.log({k: history[k][(i // config.temp_adj_step) - 1], 'batch': step})

                        elif str(k).startswith('difference_clipped_beta_indx_'):
                            wandb.log({k: history[k][(i // config.temp_adj_step) - 1], 'batch': step})

                        elif str(k).startswith('beta_'):
                            wandb.log({k: history[k][(i // config.temp_adj_step) - 1], 'batch': step})

                    if str(k).startswith('hp_'):
                        wandb.log({k: history[k][(i // config.temp_adj_step)], 'batch': step})

        for j in range(config.n_replicas):
            wandb.log({f'replica_{j}_{config.hp_to_swap}': history[j][config.hp_to_swap][i], 'batch': step} )
            wandb.log({f'exchange_loss_{j}': history[f'loss_{j}'][i], 'batch': step} )

    if swap:
        wandb.log({'num of exchange attempts': len(history['proba'])})
        wandb.log({'num of exchanges': history['swaped'].count(1)})

def log_exchange_data_mh_temp_adj_log_all_probas(history, config, swap):
    for i, step in enumerate(history['step']):
        if swap:
            wandb.log({'exchange probas': history['proba'][i], 'batch': step} )
            wandb.log({'acceptance ratio': history['accept_ratio'][i], 'batch': step} )
            wandb.log({'delta': history['delta'][i], 'batch': step} )
            wandb.log({'exchange deltas for all t': history['all_deltas'][i], 'batch': step} )
            wandb.log({'exchange_pair': history['exchange_pair'][i], 'batch': step} )
            wandb.log({'swaped': history['swaped'][i], 'batch': step} )
            wandb.log({'num_misordered_temp': history['num_misordered_temp'][i], 'batch': step} )

            # if i % config.temp_adj_step == 0 and step != 0:  #log history of proposed hp values
            #     for k in history:
            #         if str(k).startswith('hp_v_'):
            #             wandb.log({f'adj_value_for_{k}': history[k][(i // config.temp_adj_step) - 1], 'batch': step})
            if i % config.temp_adj_step == 0:  # log history of proposed hp values
                for k in history:
                    if step != 0:
                        if str(k).startswith('difference_beta_indx_'):
                            wandb.log({k: history[k][(i // config.temp_adj_step) - 1], 'batch': step})

                        elif str(k).startswith('difference_clipped_beta_indx_'):
                            wandb.log({k: history[k][(i // config.temp_adj_step) - 1], 'batch': step})

                        elif str(k).startswith('beta_'):
                            wandb.log({k: history[k][(i // config.temp_adj_step) - 1], 'batch': step})

                    if str(k).startswith('hp_'):
                        wandb.log({k: history[k][(i // config.temp_adj_step)], 'batch': step})

        for j in range(config.n_replicas):
            wandb.log({f'replica_{j}_{config.hp_to_swap}': history[j][config.hp_to_swap][i], 'batch': step} )
            wandb.log({f'exchange_loss_{j}': history[f'loss_{j}'][i], 'batch': step} )
        try:
            wandb.log({f'num_rear': history[f'num_rear'][i], 'batch': step})
        except KeyError:
            continue

    if swap:
        wandb.log({'num of exchange attempts': len(history['proba'])})
        wandb.log({'num of exchanges': history['swaped'].count(1)})



def log_exchange_data_pbt(history, config):
    for i, step in enumerate(history['step']):
        wandb.log({'optimal replica': history['optimal_replica'][i], 'batch': step})
        for j in range(config.n_replicas):
            wandb.log({f'replica_{j}_{config.hp_to_swap}': history[j][config.hp_to_swap][i], 'batch': step})
            wandb.log({f'exchange_loss_{j}': history[f'loss_{j}'][i], 'batch': step})


def assign_up_down_labels(history, hp_range, n_replicas, hp_to_swap):
    min_p, max_p = hp_range
    labels = {}
    for j in range(n_replicas):
        labels[j] = []
        curr = None
        for p in history[j][hp_to_swap]:
            if not curr:
                if p == max_p:
                    curr = 'down'
                elif p == min_p:
                    curr = 'up'
            else:
                if p == max_p and labels[j][-1] == 'up':
                    curr = 'down'
                elif p == min_p and labels[j][-1] == 'down':
                    curr = 'up'
            labels[j].append(curr)
    return labels


def calc_up_down_ratio(history, labels, hp_range, n_replicas, hp_to_swap):
    min_p, max_p = hp_range
    hp = np.array([history[j][hp_to_swap] for j in range(n_replicas)])
    ups_and_downs = {t: [] for t in np.linspace(min_p, max_p, n_replicas)}
    ups_and_downs[min_p].append('up')
    ups_and_downs[max_p].append('down')
    for i, exchange_pair in enumerate(history['exchange_pair']):
        if history['swap'][i]:
            ups_and_downs[hp[exchange_pair[0], i]].append(labels[exchange_pair[0]][i])
            ups_and_downs[hp[exchange_pair[1], i]].append(labels[exchange_pair[1]][i])
    ratios = {t: v.count('up') / (v.count('down') + v.count('up')) for t, v in ups_and_downs.items()}
    return ratios


def plot_up_down_ratios(ratios):
    plt.plot(sorted(ratios.keys()), [ratios[k] for k in sorted(ratios.keys())], '-o')
    plt.xlabel('T value')
    plt.ylabel('f(T)')
    plt.show()

def calc_num_temp_replicas_visited(history, num_replicas, replica_order):
    if replica_order:
        temp_visits = {r: [t] for t, r in enumerate(replica_order)}
    else:
        temp_visits = {r: [r] for r in range(num_replicas)}
    for i, swap in enumerate(history['swaped']):
        if swap:
            exch_pair = history['exchange_pair'][i]
            temp_visits[exch_pair[1]].append(temp_visits[exch_pair[1]][-1] + 1)
            temp_visits[exch_pair[0]].append(temp_visits[exch_pair[0]][-1] - 1)
    avg_num_of_visits = sum([len(set(t)) for t in temp_visits.values()]) / num_replicas
    frac_repl_visited_all_temp = sum([1 if len(set(t)) == num_replicas else 0 for t in temp_visits.values()]) / num_replicas
    return avg_num_of_visits, frac_repl_visited_all_temp












# def augment_image(img):
#     transform = A.Compose([A.HorizontalFlip(),
#                            A.PadIfNeeded(min_height=img.shape[0] + 8, min_width=img.shape[1] + 8),
#                            A.RandomCrop(height=img.shape[0], width=img.shape[1])
#                            ]
#                           )
#     tr_img = transform(image=img)['image']
#     return tr_img






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