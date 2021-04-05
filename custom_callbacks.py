import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage.filters import gaussian_filter1d
# import albumentations as A
import tensorflow as tf
import wandb
import functools
from utils import numpy_ewma_vectorized
from deep_tempering.callbacks import BaseExchangeCallback


class LogExchangeLossesCallback(BaseExchangeCallback):
    def __init__(self, exchange_data, swap_step, burn_in=None, **kwargs):
        super(LogExchangeLossesCallback, self).__init__(exchange_data, swap_step, burn_in)
        self.hp_to_swap = kwargs.get('hp_to_swap', None)

    def exchange(self, **kwargs):
        if self.model.global_step < 25000:
            if (25000 - self.model.global_step) <= 100:
                # print('gl st ', str(self.model.global_step))
                losses = self.evaluate_exchange_losses()
                super().log_exchange_metrics(losses)
        elif (4000 - ((self.model.global_step-25000) % 4000)) <= 100:
            # print('global step: ', str(self.model.global_step))
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


class MetropolisExchangeOneHPCallbackLogAllProbas(BaseExchangeCallback):
    """Exchanges of hyperparameters based on Metropolis acceptance criteria."""
    def __init__(self, exchange_data, hp_to_swap, swap_step=1, burn_in=1, coeff=1.):
        super(MetropolisExchangeOneHPCallbackLogAllProbas, self).__init__(exchange_data, swap_step, burn_in)
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

        if 'dropout' in hpname:
            betas = [(1. - hp) / hp for hp in hyperparams]
        else:
            betas = [1. / hp for hp in hyperparams]

        deltas = [self.coeff * (losses[i] - losses[i - 1]) * (betas[i] - betas[i - 1]) for i in range(1, len(betas))]
        probas = [min(np.exp(d), 1.) for d in deltas]

        i = exchange_pair
        j = exchange_pair - 1

        delta = deltas[exchange_pair - 1]
        proba = probas[exchange_pair - 1]

        proba=1

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
                                     all_deltas=deltas,
                                     hpname=hpname,
                                     swaped=swaped,
                                     accept_ratio=accpt_ratio,
                                     delta=delta,
                                     exchange_pair=[replicas_ids[i], replicas_ids[j]])

class PBTExchangeTruncationSelectionCallback(BaseExchangeCallback):
    """Exchanges of parameters based on PBT scheduling.

    See: Population Based Training of Neural Networks
         https://arxiv.org/abs/1711.09846
    NOTES:
      * Replica/Worker and Ensemble/Population are used interchangeably
        in the code and docs.
      * `exploit()` and `explore()` methods correspond to the ones in the
        original paper, except that perform the actions for the entire
        population (and not for single individual replica).
    """

    def __init__(self,
                 exchange_data,
                 swap_step,
                 burn_in=None,
                 explore_weights=False,
                 explore_hyperparams=True,
                 weight_dist_fn=None,
                 hyperparams_dist=None):
        """Instantiates a new `PBTExchangeCallback` instance.

        Args:
          weight_dist_fn: A function that given shape returns numpy array
            of random values that are added to the to the weights. E.g.
            `weight_dist_fn = functools.partial(np.random.normal, 0, 0.1)`
          hyperparams_dist: A dictionary that maps hyperpamater name to a
            function that returns random value by which the respective
            hyperparameter is perturbed. For example:
        """
        self.should_explore_weights = explore_weights
        self.should_explore_hyperparams = explore_hyperparams
        self.weight_dist_fn = (weight_dist_fn
                               or functools.partial(np.random.normal, 0, 0.1))
        self.hyperparams_dist = hyperparams_dist

        super(PBTExchangeTruncationSelectionCallback, self).__init__(exchange_data, swap_step, burn_in)

    def exploit_and_explore(self, **kwargs):
        """Decides whether  the worker should abandon the current solution.

        Given performance of the whole population, can decide whether the
        worker should abandon the current solution and instead focus on a
        more promising one; and `explore`, which given the current solution
        and hyperparameters proposes new ones to better explore the
        solution space.

        `exploit` could replace the current weights with the weights that
        have the highest recorded performance in the rest of the
        population, and `explore` could randomly perturb the
        hyperparameters with noise.

        In short, copies weights and hyperparams from optimal replica and
        perturbs them.
        """
        # `test_losses` is used for testing to verify the logic.
        losses = kwargs.get('test_losses', None) or self.evaluate_exchange_losses()
        optimal_replica_id = np.argmin(losses)

        replicas_ids_sorted_by_performance = np.argsort(losses)[::-1]
        twenty_p = int(np.ceil(self.model.n_replicas * 0.2))
        bottom_20 = replicas_ids_sorted_by_performance[:twenty_p]
        top_20 = replicas_ids_sorted_by_performance[-twenty_p:]
        optimal_weights = self.model.models[optimal_replica_id].trainable_variables

        # copy vars
        for rid in bottom_20:
            if not tf.executing_eagerly():
                src_replica_id = np.random.choice(top_20)
                self.copy_weights(src_replica_id, rid)

                if self.should_explore_weights:
                    self.explore_weights(rid)
                if self.should_explore_hyperparams:
                    # copy hparams and perturb
                    self.model.hpspace.copy_hyperparams(src_replica_id, rid)
                    self.explore_hyperparams(rid)

            else:
                raise NotImplementedError()

        super().log_exchange_metrics(losses, optimal_replica=optimal_replica_id)

    def copy_weights(self, src_replica, dst_replica):
        """Copies variables from `src_replica` to `dst_replica`."""
        # print('copy_weights ---> src_replica:', src_replica, ', dst_replica:', dst_replica)
        src_model = self.model.models[src_replica]
        dst_model = self.model.models[dst_replica]
        src_vars = src_model.trainable_variables
        dst_vars = dst_model.trainable_variables
        sess = tf.compat.v1.keras.backend.get_session()
        for vsrc, vdst in zip(src_vars, dst_vars):
            np_vsrc = vsrc.eval(session=sess)
            vdst.load(np_vsrc, session=sess)

    def explore_weights(self, replica_id):
        """Perturbs weights of `replica_id` with noise.

        Args:
          replica_id: The ID of replica that needs to be perturbed.
        """
        weight_dist_fn = (self.weight_dist_fn
                          or functools.partial(np.random.normal, 0, 0.1))

        sess = tf.compat.v1.keras.backend.get_session()
        model = self.model.models[replica_id]
        for w in model.trainable_variables:
            shape = w.get_shape().as_list()
            value = sess.run(w)
            perturbed_value = value + self.weight_dist_fn(shape)
            w.load(perturbed_value, session=sess)

    def explore_hyperparams(self, replica_id):
        """Perturbs hyperparams of `replica_id`."""
        if self.hyperparams_dist is not None:
            for hpname, dist in self.hyperparams_dist.items():
                self.model.hpspace.perturb_hyperparams(
                    replica_id, hpname, dist)

    def copy_hyperparams(self, src_replica, dst_replica):
        """Copies variables from `src_replica` to `dst_replica`."""
        hps = self.model.hpspace
        for hpname in hps.hpspace[0]:
            hps.hpspace[dst_replica][hpname] = hps.hpspace[src_replica][hpname]

    def exchange(self, *args, **kwargs):
        self.exploit_and_explore(*args, **kwargs)



class MetropolisExchangeTempAdjustmentCallback(BaseExchangeCallback):
    """Exchanges of hyperparameters based on Metropolis acceptance criteria."""
    def __init__(self, exchange_data, hp_to_swap, swap_step=1, burn_in=1, coeff=1., temp_adj_step=5):
        super(MetropolisExchangeTempAdjustmentCallback, self).__init__(exchange_data, swap_step, burn_in)
        self.coeff = coeff
        self.hpname = hp_to_swap
        self.temp_adj_step = temp_adj_step

    def calc_adj_value(self, beta_old, beta_new, beta_pr, beta_next):
        diff = beta_old - beta_new
        if diff < 0: #beta_old < beta_new, beta_new is between beta_old and beta_pr
            adj_value = min(abs(diff), abs(beta_pr - beta_old) / 2) #adj value can't be bigger than half dist btw beta_prev and beta_old
            # adj_value = min(abs(diff), abs(beta_pr - beta_old) - abs((beta_pr - beta_old) / 10))
            beta_adj = beta_old + adj_value
        elif diff > 0:
            adj_value = min(abs(diff), abs(beta_next - beta_old) / 2)
            # adj_value = min(abs(diff), abs(beta_next - beta_old) - abs((beta_next - beta_old) / 10))

            beta_adj = beta_old - adj_value
        else:
            beta_adj = beta_new
        return beta_adj, diff, beta_old - beta_adj


    def adjust_temperatures(self, temp_adj_count):
        n_replicas = self.model.n_replicas
        hp_name = self.hpname
        hp = self.ordered_hyperparams

        losses_history = [self.exchange_logs[f'loss_{i}'][-self.temp_adj_step:] for i in range(n_replicas)] #TEMP, how many of the last losses to take?
        hp_values = [h[1] for h in hp[hp_name]]
        replicas_ids = [h[0] for h in hp[hp_name]]
        losses_history_by_temp = {hp_value: [] for hp_value in hp_values}  #review case: if we add new hp value because we adj them wi

        for r in range(n_replicas):
            for step, hp_value in enumerate(self.exchange_logs[r][hp_name][-self.temp_adj_step:]):
                losses_history_by_temp[hp_value].append(losses_history[r][step])

        losses_history_by_temp = {k: np.mean(v) for k, v in losses_history_by_temp.items()}  #for now, use a simple mean ToDo: add exponential avg


        if 'dropout' in hp_name:
            betas = [(1. - hp_value) / hp_value for hp_value in losses_history_by_temp]
        else:
            betas = [1. / hp_value for hp_value in losses_history_by_temp] # our betas aren't equidistant
                                                                            #
        if temp_adj_count % 2: #
            start_indx = 1
        else:
            start_indx = 2

        # adj_hp_values = {}
        diff_btw_betas = {}
        diff_btw_betas_clipped = {}
        betas_to_log = {i: b for i, b in enumerate(betas)}
        for i in range(start_indx, n_replicas - 1, 2): #don't change the top and bottom temp
            beta_pr, beta_next = betas[i-1], betas[i+1]
            loss_pr, loss_i, loss_next = losses_history_by_temp[hp_values[i-1]], losses_history_by_temp[hp_values[i]], losses_history_by_temp[hp_values[i+1]]

            d = loss_pr - loss_next
            n = (beta_pr * loss_pr - beta_next * loss_next - loss_i * (beta_pr - beta_next))

            beta_new = n / d

            beta_adj = ((betas[i] + beta_new) / 2.) # new value should be between beta_pr and beta_next

            beta_adj, diff, diff_clipped = self.calc_adj_value(betas[i], beta_adj, beta_pr, beta_next)
            if 'dropout' in hp_name:
                hp_value_new = 1. / (beta_adj + 1.)
            else:
                hp_value_new = 1. / beta_adj
            betas_to_log[i] = beta_adj


            # adj_hp_values[np.round(hp_values[i], 5)] = hp_value_new
            diff_btw_betas[i] = diff
            diff_btw_betas_clipped[i] = diff_clipped

            self.model.hpspace.set_hyperparams(hp_value_new, hp_name, replicas_ids[i])

        # for hp_v in hp_values:
        #     if not np.round(hp_v, 5) in adj_hp_values:
        #         adj_hp_values[np.round(hp_v, 5)] = hp_v
        for i in range(n_replicas):
            if i not in diff_btw_betas:
                diff_btw_betas[i] = 0
            if i not in diff_btw_betas_clipped:
                diff_btw_betas_clipped[i] = 0

        return diff_btw_betas, diff_btw_betas_clipped, betas_to_log




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

        if self.model.global_step == 0:
            for i, hp in enumerate(hyperparams):
                self.exchange_logs[f'hp_value_{i}'] = [hp]

        if (len(self.exchange_logs['swaped']) - 1) % self.temp_adj_step == 0 and len(self.exchange_logs['swaped']) > 1: #adjust after swap??? adjust only after a certain number of swaps or count no swaps as well?
            for i, hp in enumerate(self.ordered_hyperparams[hpname]): #some of them were changed in adj_temperatures
                self.exchange_logs[f'hp_value_{i}'].append(hp[1])
            diff_btw_betas, diff_btw_betas_clipped, betas_to_log = self.adjust_temperatures((len(self.exchange_logs['swaped']) - 1) // self.temp_adj_step) #subtract 1 bc of the call to the exchange at the beginning of the training
            for k, v in diff_btw_betas.items():
                if not f'difference_beta_indx_{k}' in self.exchange_logs:
                    self.exchange_logs[f'difference_beta_indx_{k}'] = [v]
                else:
                    self.exchange_logs[f'difference_beta_indx_{k}'].append(v)

            for k, v in diff_btw_betas_clipped.items():
                if not f'difference_clipped_beta_indx_{k}' in self.exchange_logs:
                    self.exchange_logs[f'difference_clipped_beta_indx_{k}'] = [v]
                else:
                    self.exchange_logs[f'difference_clipped_beta_indx_{k}'].append(v)

            for k, v in betas_to_log.items():
                if not f'beta_{k}' in self.exchange_logs:
                    self.exchange_logs[f'beta_{k}'] = [v]
                else:
                    self.exchange_logs[f'beta_{k}'].append(v)


class MetropolisExchangeTempAdjustmentCallbackLogAllProbas(BaseExchangeCallback):
    """Exchanges of hyperparameters based on Metropolis acceptance criteria."""

    def __init__(self, exchange_data, hp_to_swap, swap_step=1, burn_in=1, coeff=1., temp_adj_step=5):
        super(MetropolisExchangeTempAdjustmentCallbackLogAllProbas, self).__init__(exchange_data, swap_step, burn_in)
        self.coeff = coeff
        self.hpname = hp_to_swap
        self.temp_adj_step = temp_adj_step

    def calc_adj_value(self, beta_old, beta_new, beta_pr, beta_next):
        diff = beta_old - beta_new
        if diff < 0:  # beta_old < beta_new, beta_new is between beta_old and beta_pr
            adj_value = min(abs(diff), abs(beta_pr - beta_old) / 2) #adj value can't be bigger than half dist btw beta_prev and beta_old
            # adj_value = min(abs(diff), abs(beta_pr - beta_old) - abs((beta_pr - beta_old) / 20))
            beta_adj = beta_old + adj_value
        elif diff > 0:
            adj_value = min(abs(diff), abs(beta_next - beta_old) / 2)
            # adj_value = min(abs(diff), abs(beta_next - beta_old) - abs((beta_next - beta_old) / 20))

            beta_adj = beta_old - adj_value
        else:
            beta_adj = beta_new
        return beta_adj, diff, beta_old - beta_adj

    def get_losses_per_temp_every_exchange_step(self):
        n_replicas = self.model.n_replicas
        hp_name = self.hpname
        hp = self.ordered_hyperparams
        losses_history = [self.exchange_logs[f'loss_{i}'][-self.temp_adj_step:] for i in
                          range(n_replicas)]  # TEMP, how many of the last losses to take?
        hp_values = [h[1] for h in hp[hp_name]]
        losses_history_by_temp = {hp_value: [] for hp_value in
                                  hp_values}  # review case: if we add new hp value because we adj them wi

        for r in range(n_replicas):
            for step, hp_value in enumerate(self.exchange_logs[r][hp_name][-self.temp_adj_step:]):
                losses_history_by_temp[hp_value].append(losses_history[r][step])

        losses_history_by_temp = {k: numpy_ewma_vectorized(np.array(v), len(v) - 2)[-1] for k, v in
                                  losses_history_by_temp.items()}  # for now, use a simple mean

        return losses_history_by_temp

    # def get_losses_per_temp_last_n_train_step(self):
    #     steps_to_look_back = 100
    #     n_replicas = self.model.n_replicas
    #     hp_name = self.hpname
    #     hp = self.ordered_hyperparams
    #     hp_values = [h[1] for h in hp[hp_name]]
    #
    #     repl2t = {r[0]: t for t, r in enumerate(hp[hp_name])}
    #     losses_history = [self.per_batch_losses_callback.exchange_logs[f'loss_{i}'][-steps_to_look_back:][::-1] for i in
    #                       range(n_replicas)]
    #     # print(len(losses_history))
    #
    #     losses_history_by_temp = {i: [] for i in range(n_replicas)}
    #
    #     n = 1
    #     for s in range(steps_to_look_back):
    #         if s % self.swap_step == 0:
    #             if self.exchange_logs['swaped'][-n]:
    #                 exch_pair = self.exchange_logs['exchange_pair'][-n]
    #                 repl2t[exch_pair[0]], repl2t[exch_pair[1]] = repl2t[exch_pair[1]], repl2t[exch_pair[0]]
    #             n += 1
    #         for r in range(n_replicas):
    #             losses_history_by_temp[repl2t[r]].append(losses_history[r][s])
    #
    #     losses_history_by_temp = {hp_values[k]: np.mean(v) for k, v in
    #                               losses_history_by_temp.items()}
    #
    #     return losses_history_by_temp

    def adjust_temperatures(self, temp_adj_count):
        print('adjusting')
        n_replicas = self.model.n_replicas
        hp_name = self.hpname
        hp = self.ordered_hyperparams
        hp_values = [h[1] for h in hp[hp_name]]
        replicas_ids = [h[0] for h in hp[hp_name]]

        losses_history_by_temp = self.get_losses_per_temp_every_exchange_step()

        # print(len(losses_history_by_temp.))

        if 'dropout' in hp_name:
            betas = [(1. - hp_value) / hp_value for hp_value in losses_history_by_temp]
        else:
            betas = [1. / hp_value for hp_value in losses_history_by_temp]  # our betas aren't equidistant
            #
        if temp_adj_count % 2:  #
            start_indx = 1
        else:
            start_indx = 2

        # adj_hp_values = {}
        diff_btw_betas = {}
        diff_btw_betas_clipped = {}
        betas_to_log = {i: b for i, b in enumerate(betas)}
        for i in range(start_indx, n_replicas - 1, 2):  # don't change the top and bottom temp
            beta_pr, beta_next = betas[i - 1], betas[i + 1]
            loss_pr, loss_i, loss_next = losses_history_by_temp[hp_values[i - 1]], losses_history_by_temp[hp_values[i]], \
                                         losses_history_by_temp[hp_values[i + 1]]

            d = loss_pr - loss_next
            n = (beta_pr * loss_pr - beta_next * loss_next - loss_i * (beta_pr - beta_next))

            beta_new = n / d

            beta_adj = ((betas[i] + beta_new) / 2.)  # new value should be between beta_pr and beta_next

            beta_adj, diff, diff_clipped = self.calc_adj_value(betas[i], beta_adj, beta_pr, beta_next)
            hp_value_new = 1. / beta_adj
            betas_to_log[i] = beta_adj

            # adj_hp_values[np.round(hp_values[i], 5)] = hp_value_new
            diff_btw_betas[i] = diff
            diff_btw_betas_clipped[i] = diff_clipped

            self.model.hpspace.set_hyperparams(hp_value_new, hp_name, replicas_ids[i])

        # for hp_v in hp_values:
        #     if not np.round(hp_v, 5) in adj_hp_values:
        #         adj_hp_values[np.round(hp_v, 5)] = hp_v
        for i in range(n_replicas):
            if i not in diff_btw_betas:
                diff_btw_betas[i] = 0
            if i not in diff_btw_betas_clipped:
                diff_btw_betas_clipped[i] = 0

        return diff_btw_betas, diff_btw_betas_clipped, betas_to_log

    def exchange(self, **kwargs):
        """Exchanges hyperparameters between adjacent replicas.

        This function is called once on the beginning of training to
        log initial values of hyperparameters and then it is called
        every `swap_step` steps.
        """ # pick random hyperparameter to exchange
        hp = self.ordered_hyperparams
        hpname = self.hpname
        # pick random replica pair to exchange
        n_replicas = self.model.n_replicas
        exchange_pair = kwargs.get('exchange_pair', np.random.randint(1, n_replicas))

        losses = self.evaluate_exchange_losses()

        hyperparams = [h[1] for h in hp[hpname]]
        replicas_ids = [h[0] for h in hp[hpname]]

        if 'dropout' in hpname:
            betas = [(1. - hp) / hp for hp in hyperparams]
        else:
            betas = [1. / hp for hp in hyperparams]

        deltas = [self.coeff * (losses[i] - losses[i - 1]) * (betas[i] - betas[i - 1]) for i in range(1, len(betas))]
        probas = [min(np.exp(d), 1.) for d in deltas]

        i = exchange_pair
        j = exchange_pair - 1

        delta = deltas[exchange_pair - 1]
        proba = probas[exchange_pair - 1]

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
                                     all_deltas=deltas,
                                     hpname=hpname,
                                     swaped=swaped,
                                     accept_ratio=accpt_ratio,
                                     delta=delta,
                                     exchange_pair=[replicas_ids[i], replicas_ids[j]])


        if self.model.global_step == 0:
            for i, hp in enumerate(hyperparams):
                self.exchange_logs[f'hp_value_{i}'] = [hp]

        if (len(self.exchange_logs['swaped']) - 1) % self.temp_adj_step == 0 and len(self.exchange_logs[
                                                                                         'swaped']) > 1:  # adjust after swap??? adjust only after a certain number of swaps or count no swaps as well?
            for i, hp in enumerate(self.ordered_hyperparams[hpname]):  # some of them were changed in adj_temperatures
                self.exchange_logs[f'hp_value_{i}'].append(hp[1])
            diff_btw_betas, diff_btw_betas_clipped, betas_to_log = self.adjust_temperatures((len(self.exchange_logs[
                                                                                                     'swaped']) - 1) // self.temp_adj_step)  # subtract 1 bc of the call to the exchange at the beginning of the training
            for k, v in diff_btw_betas.items():
                if not f'difference_beta_indx_{k}' in self.exchange_logs:
                    self.exchange_logs[f'difference_beta_indx_{k}'] = [v]
                else:
                    self.exchange_logs[f'difference_beta_indx_{k}'].append(v)

            for k, v in diff_btw_betas_clipped.items():
                if not f'difference_clipped_beta_indx_{k}' in self.exchange_logs:
                    self.exchange_logs[f'difference_clipped_beta_indx_{k}'] = [v]
                else:
                    self.exchange_logs[f'difference_clipped_beta_indx_{k}'].append(v)

            for k, v in betas_to_log.items():
                if not f'beta_{k}' in self.exchange_logs:
                    self.exchange_logs[f'beta_{k}'] = [v]
                else:
                    self.exchange_logs[f'beta_{k}'].append(v)







