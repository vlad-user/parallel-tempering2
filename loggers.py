import numpy as np
import tensorflow as tf
import wandb
from keras.datasets import cifar10
from keras.utils import np_utils
from read_datasets import get_emnist_letters
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as shuffle_dataset


def prepare_data(args):
    if args.dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print(x_test.shape)

    elif args.dataset_name == 'emnist':
        x_train, y_train, x_test, y_test = get_emnist_letters()

        x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)),
           mode='constant', constant_values=0)
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)),
                         mode='constant', constant_values=0)
        y_train = np.int32(y_train) - 1
        y_test = np.int32(y_test) - 1


    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # mean = np.array([np.mean(x_train[..., c]) for c in range(3)])
    # std = np.array([np.std(x_train[..., c]) for c in range(3)])
    #
    # for x in [x_train, x_test]:
    #     x[..., 0] -= mean[0]
    #     x[..., 1] -= mean[1]
    #     x[..., 2] -= mean[2]
    #     x[..., 0] /= (std[0] + 1e-7)
    #     x[..., 1] /= (std[1] + 1e-7)
    #     x[..., 2] /= (std[2] + 1e-7)

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    # m = np.mean(x_train, axis=(1,2), keepdims=True)
    # std = np.std(x_train, axis=(1,2), keepdims=True)
    # x_train = (x_train - m) / std
    # x_test = (x_test - m[:len(x_test)]) / std[:len(x_test)]

    y_train = np_utils.to_categorical(y_train, num_classes=np.max(y_train)+1)

    y_test = np_utils.to_categorical(y_test, num_classes=np.max(y_test)+1)

    x_train, y_train = shuffle_dataset(x_train, y_train, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    # x_val, y_val = deepcopy(x_test), deepcopy(y_test)
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
