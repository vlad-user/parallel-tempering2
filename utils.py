import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage.filters import gaussian_filter1d
# import albumentations as A
import tensorflow as tf
import wandb
from deep_tempering.callbacks import BaseExchangeCallback


class LogExchangeLossesCallback(BaseExchangeCallback):
    def __init__(self, exchange_data, swap_step, burn_in=None, **kwargs):
        super(LogExchangeLossesCallback, self).__init__(exchange_data, swap_step, burn_in)
        self.hp_to_swap = kwargs.get('hp_to_swap', None)

    def exchange(self, **kwargs):
        losses = self.evaluate_exchange_losses()
        super().log_exchange_metrics(losses)


class MetropolisExchangeOneHPCallback(BaseExchangeCallback):
  """Exchanges of hyperparameters based on Metropolis acceptance criteria."""
  def __init__(self, exchange_data, hp_to_swap, swap_step=1, burn_in=1, coeff=1.):
    super(MetropolisExchangeOneHPCallback, self).__init__(exchange_data, swap_step, burn_in)
    self.coeff = coeff
    self.hpname = hp_to_swap

  def exchange(self, **kwargs):
    """Exchanges hyperparameters between adjacent replicas.

    This function is called once on the beginning of training to
    log initial values of hyperparameters and then it is called
    every `swap_step` steps.
    """
    # pick random hyperparameter to exchange
    hp = self.ordered_hyperparams
    hpname = self.hpname
    # pick random replica pair to exchange
    n_replicas = self.model.n_replicas
    exchange_pair = kwargs.get('exchange_pair', np.random.randint(1, n_replicas))

    losses = self.evaluate_exchange_losses()

    hyperparams = [h[1] for h in hp[hpname]]
    replicas_ids = [h[0] for h in hp[hpname]]

    i = exchange_pair
    j = exchange_pair - 1

    # compute betas
    if 'dropout' in hpname:
      beta_i = (1. - hyperparams[i]) / hyperparams[i]
      beta_j = (1. - hyperparams[j]) / hyperparams[j]
    else:
      # learning rate
      beta_i = 1. / hyperparams[i]
      beta_j = 1. / hyperparams[j]

    # beta_i - beta_j is expected to be negative
    delta = self.coeff * (losses[i] - losses[j]) * (beta_i - beta_j)
    proba = min(np.exp(delta), 1.)

    if np.random.uniform() < proba:
      swaped = 1
      self.model.hpspace.swap_between(replicas_ids[i], replicas_ids[j], hpname)
    else:
      swaped = 0

    if getattr(self, 'exchange_logs', None):
      accpt_ratio = (self.exchange_logs['swaped'].count(1) + swaped) / \
                    (len(self.exchange_logs['proba']) + 1)
    else:
      accpt_ratio = swaped

    super().log_exchange_metrics(losses,
                                 proba=proba,
                                 hpname=hpname,
                                 swaped=swaped,
                                 accept_ratio=accpt_ratio,
                                 delta=delta,
                                 exchange_pair=[replicas_ids[i], replicas_ids[j]])


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


def log_exchange_data_pbt(history, config):
    for i, step in enumerate(history['step']):
        wandb.log({'optimal replica': history['optimal_replica'][i]}, step=step)
    val_acc = np.array([history[f'val_acc_{i}'] for i in range(config.n_replicas)])
    wandb.log({'best val acc, # of replica, step': [np.round(np.max(val_acc), 3), np.argmax(np.max(val_acc, axis=1)),
                                                    np.argmax(val_acc) % val_acc.shape[1]]})


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