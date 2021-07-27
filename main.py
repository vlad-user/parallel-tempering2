import argparse
from model_builders import lenet5_emnist_builder, lenet5_cifar10_builder, lenet5_cifar10_with_augmentation_builder, lenet5_cifar10_same_init_builder, resnet50_cifar10, densenet121_cifar10, densenet121_dropout_cifar10_aug, resnet_v2_20_cifar10, resnet20_v1_cifar10, resnet56_v1_cifar, resnet121_v1_cifar
from utils import *
from custom_callbacks import *
import deep_tempering as dt
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler

import time


import os
import random
import wandb


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--notes", type=str, default=' ')


    parser.add_argument("--model_name", type=str, default='lenet5')
    parser.add_argument("--dataset_name", type=str, default='cifar10')
    parser.add_argument("--model_builder", type=str, default='lenet5_cifar10_same_init_builder')

    parser.add_argument("--hp_to_swap", type=str, default='learning_rate')
    parser.add_argument("--n_replicas", type=int, default=8)
    parser.add_argument("--exchange_type", type=str, choices=['no_swap',
                                                              'swap', 'swap_adj', 'swap_sort_adj', 'swap_rear', 'swap_rear_adj', 'swap_sort', 'swap_pbt'])

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--proba_coeff_C", type=float, default=3.)
    parser.add_argument("--train_data_size", type=int, default=45000)
    parser.add_argument("--lr_min", type=float, default=0.01)
    parser.add_argument("--lr_max", type=float, default=0.02)
    parser.add_argument("--dropout_rate_min", type=float, default=0.4)
    parser.add_argument("--dropout_rate_max", type=float, default=0.4)
    parser.add_argument('--pbt_std', type=float, default=0.001)

    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--swap_step", type=int, default=400)
    parser.add_argument("--temp_adj_step", type=int, default=10)
    parser.add_argument("--n_prev_eval_steps", type=int, default=8)

    parser.add_argument("--burn_in_hp", type=int, default=20000)
    parser.add_argument("--burn_in_rearranger", type=int, default=21200)
    parser.add_argument("--burn_in_exchanger", type=int, default=25000)
    parser.add_argument("--do_swap", type=bool, default=True)

    parser.add_argument('--use_ensemble_model', action='store_true')

    args = parser.parse_args()
    assert args.exchange_type

    if args.exchange_type == 'no_swap':
        args.do_swap = False

    return args


def init_clbks(args,  x_val, y_val):

    weights_sort_clbk = WeightsSortCallback(exchange_data=(x_val, y_val),
                                            hp_to_swap=args.hp_to_swap,
                                            swap_step=400,
                                            burn_in=args.burn_in_rearranger,
                                            n_prev_eval_steps=args.n_prev_eval_steps,
                                            n_replicas=args.n_replicas
                                            )

    all_losses_clbk = LogExchangeLossesCallback(exchange_data=(x_val, y_val),
                                                hp_to_swap=args.hp_to_swap,
                                                swap_step=400,
                                                burn_in=0,
                                                n_prev_eval_steps=args.n_prev_eval_steps,
                                                do_swap=args.do_swap,
                                                weights_rear_clbk=weights_sort_clbk
                                                )

    clbks = [all_losses_clbk]

    if args.exchange_type == 'swap':
        exch_clbk = MetropolisExchangeOneHPCallbackLogAllProbas(
                              exchange_data=(x_val, y_val),
                              hp_to_swap=args.hp_to_swap,
                              swap_step=args.swap_step,
                              burn_in=args.burn_in_exchanger,
                              coeff=args.proba_coeff_C,
                              n_prev_eval_steps=args.n_prev_eval_steps,
                              weights_sort_clbk=None
                              )
        clbks.append(exch_clbk)

    elif args.exchange_type == 'swap_adj':
        exch_clbk = MetropolisExchangeTempAdjustmentCallbackLogAllProbas(
            all_losses_clbk=all_losses_clbk,
            exchange_data=(x_val, y_val),
            hp_to_swap=args.hp_to_swap,
            swap_step=args.swap_step,
            burn_in=args.burn_in_exchanger,
            coeff=args.proba_coeff_C,
            temp_adj_step=args.temp_adj_step,
            n_prev_eval_steps=args.n_prev_eval_steps,
            weights_sort_clbk=None

        )
        clbks.append(exch_clbk)

    elif args.exchange_type == 'swap_sort_adj':
        exch_clbk = MetropolisExchangeTempAdjustmentCallbackLogAllProbas(
            all_losses_clbk=all_losses_clbk,
            exchange_data=(x_val, y_val),
            hp_to_swap=args.hp_to_swap,
            swap_step=args.swap_step,
            burn_in=args.burn_in_exchanger,
            coeff=args.proba_coeff_C,
            temp_adj_step=args.temp_adj_step,
            n_prev_eval_steps=args.n_prev_eval_steps,
            weights_sort_clbk=weights_sort_clbk

        )
        clbks.append(exch_clbk)

        clbks.append(weights_sort_clbk)


    elif args.exchange_type == 'swap_rear_adj':
        rearanger_clbk = ReplicaRearrangerCallback(exchange_data=(x_val, y_val),
                                                   swap_step=args.swap_step,
                                                   burn_in=args.burn_in_rearranger,
                                                   n_prev_eval_steps=args.n_prev_eval_steps,
                                                   perturb_func=None
                                                   )
        exch_clbk = MetropolisExchangeTempAdjustmentCallbackLogAllProbas(
            all_losses_clbk=all_losses_clbk,
            exchange_data=(x_val, y_val),
            hp_to_swap=args.hp_to_swap,
            swap_step=args.swap_step,
            burn_in=args.burn_in_exchanger,
            coeff=args.proba_coeff_C,
            temp_adj_step=args.temp_adj_step,
            n_prev_eval_steps=args.n_prev_eval_steps,
            weights_sort_clbk=None

        )
        clbks.append(exch_clbk)

        clbks.append(rearanger_clbk)

    elif args.exchange_type == 'swap_rear':
        rearanger_clbk = ReplicaRearrangerCallback(exchange_data=(x_val, y_val),
                                                   swap_step=args.swap_step,
                                                   burn_in=args.burn_in_rearranger,
                                                   n_prev_eval_steps=args.n_prev_eval_steps,
                                                   perturb_func=None
                                                   )
        exch_clbk = MetropolisExchangeOneHPCallbackLogAllProbas(
                              exchange_data=(x_val, y_val),
                              hp_to_swap=args.hp_to_swap,
                              swap_step=args.swap_step,
                              burn_in=args.burn_in_exchanger,
                              coeff=args.proba_coeff_C,
                              n_prev_eval_steps=args.n_prev_eval_steps,
                              weights_sort_clbk=None
                              )
        clbks.append(exch_clbk)

        clbks.append(rearanger_clbk)
    
    elif args.exchange_type == 'swap_sort':
        exch_clbk = MetropolisExchangeOneHPCallbackLogAllProbas(
                              exchange_data=(x_val, y_val),
                              hp_to_swap=args.hp_to_swap,
                              swap_step=args.swap_step,
                              burn_in=args.burn_in_exchanger,
                              coeff=args.proba_coeff_C,
                              n_prev_eval_steps=args.n_prev_eval_steps,
                              weights_sort_clbk=weights_sort_clbk
                              )

        clbks.append(exch_clbk)

        clbks.append(weights_sort_clbk)

    elif args.exchange_type == 'swap_pbt':
        if args.hp_to_swap == 'learning_rate':
            hparams_dist_dict = {
              'learning_rate': lambda *x: np.random.normal(0.0, args.pbt_std),
                # 'learning_rate': lambda *x: np.random.choice([0.8, 1.2]),
              'dropout_rate': lambda *x: 0
              }
        elif args.hp_to_swap == 'dropout_rate':
            hparams_dist_dict = {
                'dropout_rate': lambda *x: np.random.normal(0.0, args.pbt_std),
                # 'learning_rate': lambda *x: np.random.choice([0.8, 1.2]),
                'learning_rate': lambda *x: 0
            }
        else:
            raise NotImplementedError('Specified hp is not valid for PBT optimization.')
        exch_clbk = PBTExchangeTruncationSelectionCallback(
                     exchange_data=(x_val, y_val),
                     swap_step=args.swap_step,
                     explore_weights=False,
                     explore_hyperparams=True,
                     burn_in=args.burn_in_exchanger,
                     hyperparams_dist=hparams_dist_dict)
        clbks.append(exch_clbk)

    
    # if args.exchange_type != 'no_swap':

    return clbks

def lr_schedule_resnet(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def lr_schedule_lenet(epoch):
    if epoch > 150:
        return 0.0001
    else:
        return 0.001


def main():
    args = init_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    model_builders = {"lenet5_cifar10_builder": lenet5_cifar10_builder, 'lenet5_emnist_builder': lenet5_emnist_builder,
                      "lenet5_cifar10_with_augmentation_builder": lenet5_cifar10_with_augmentation_builder,
                      'lenet5_cifar10_same_init_builder': lenet5_cifar10_same_init_builder, 'resnet50_cifar10_builder': resnet50_cifar10,
                      'densenet121_cifar10_builder': densenet121_cifar10, 'densenet121_dropout_cifar10_builder': densenet121_dropout_cifar10_aug,
                      'resnet20_v2_cifar10_builder': resnet_v2_20_cifar10,
                      'resnet20_v1_cifar10_builder': resnet20_v1_cifar10,
                      'resnet56_v1_cifar10_builder': resnet56_v1_cifar,
                      'resnet121_v1_cifar10_builder': resnet121_v1_cifar,
                      }

    if args.model_name.startswith('resnet'):
        hp = {1: {'learning_rate': [1e-3 for _ in range(args.n_replicas)],
                  'dropout_rate': np.linspace(args.dropout_rate_min, args.dropout_rate_max, args.n_replicas), },
              args.burn_in_hp: {'learning_rate': np.linspace(args.lr_min, args.lr_max, args.n_replicas),

                      'dropout_rate': np.linspace(args.dropout_rate_min, args.dropout_rate_max, args.n_replicas)},

            }
        lr_schedule = lr_schedule_resnet

    # if args.model_name.startswith('resnet'):
    #     hp = {1:{'learning_rate': [1e-3 for _ in range(args.n_replicas)]}, 80*390:{'learning_rate': [1e-4 for _ in range(args.n_replicas)]},
    #           120*390:{'learning_rate': [1e-5 for _ in range(args.n_replicas)]},
    #          160*390: {'learning_rate': [1e-6 for _ in range(args.n_replicas)]},
    #           180*390: {'learning_rate': [1e-3*0.5e-3 for _ in range(args.n_replicas)]}}
    #     lr_schedule = lr_schedule_resnet
    else:
        hp = {1: {'learning_rate': [0.1 for _ in range(args.n_replicas)],
                            'dropout_rate': np.linspace(args.dropout_rate_min, args.dropout_rate_max, args.n_replicas), },
                        args.burn_in_hp: {'learning_rate': np.linspace(args.lr_min, args.lr_max, args.n_replicas),

                                'dropout_rate': np.linspace(args.dropout_rate_min, args.dropout_rate_max, args.n_replicas)},
              }
        lr_schedule = lr_schedule_lenet

    wandb.init(
        project="deep-tempering",
        name=f"{args.exp_name}-{args.random_seed}",
        config=vars(args),
        notes=args.notes,
    )

    x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(args)

    assert x_train.shape[0] == args.train_data_size

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.
    session = tf.Session(config=config, )

    if args.use_ensemble_model:
        model = dt.EnsembleModel(model_builders[args.model_builder])
        model.compile(optimizer=SGD(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      n_replicas=args.n_replicas)
    else:
        model = model_builders[args.model_builder]()
        model.compile(optimizer=SGD(lr=lr_schedule(0)),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      )



    model.summary()

    clbks = init_clbks(args, x_val, y_val)

    start = time.time()

    if args.use_ensemble_model:
        history = model.fit(x_train,
                            y_train,
                            validation_data=(x_test, y_test),
                            hyper_params=hp,
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            # random_data_split_state=args.random_seed,
                            shuffle=True,
                            callbacks=clbks
                            )

        addt_losses = clbks[0].exchange_logs

        log_additional_losses(addt_losses, args, args.do_swap)
    else:
        lr_sc = LearningRateScheduler(lr_schedule)
        history = model.fit(x_train,
                            y_train,
                            validation_data=(x_test, y_test),
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            # random_data_split_state=args.random_seed,
                            shuffle=True,
                            callbacks=[lr_sc]
                            )




    end  = time.time() - start


    history = history.history
    mname = 'acc_0' if args.use_ensemble_model else 'acc'
    for step in range(len(history[mname])):
        for k in sorted(history.keys()):
            wandb.log({k: history[k][step], 'epoch': step})

    val_acc = np.array([history[f'val_acc_{i}'] for i in range(args.n_replicas)]) if args.use_ensemble_model else np.array([history[f'val_acc'] for i in range(args.n_replicas)])
    wandb.log({'best val acc, # of replica, step': [np.round(np.max(val_acc), 3), np.argmax(np.max(val_acc, axis=1)),
                                                    np.argmax(val_acc) % val_acc.shape[1]]})
    wandb.log({'sec/epoch': end / args.epochs})



    if args.exchange_type != 'no_swap':

        ex_history = clbks[1].exchange_logs

        avg_num_temp_repl_visited, frac_visited_all_temp = calc_num_temp_replicas_visited(ex_history,
                                                                                          args.n_replicas,
                                                                                          replica_order=clbks[0].weights_sort_clbk.replica_order)
        wandb.log({'avg_num_temp_repl_visited': avg_num_temp_repl_visited})
        wandb.log({'frac_of_repl_visited_all_temp': frac_visited_all_temp})

        if 'adj' in args.exchange_type:
            log_exchange_data_mh_temp_adj_log_all_probas(ex_history, args, args.do_swap)

        elif 'pbt' in args.exchange_type:
            log_exchange_data_pbt(ex_history, args)

        else:
            log_exchange_data_mh_log_all_probas(ex_history, args, args.do_swap)

if __name__ == '__main__':
    main()

