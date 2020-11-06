import wandb
from utils import *
import json
import os
api = wandb.Api()


def load_logs(run_path, metrics_to_load):
    p = os.path.join('logs', run_path.split('/')[-1] + '.json')
    print(p)
    if os.path.exists(p):
        with open(p, 'r') as fp:
            logs = json.load(fp)
            return logs

    logs = {}
    for k in metrics_to_load:
        run = api.run(run_path)
        h = run.scan_history(keys=[k])
        rows = [row for row in h]
        logs[k] = [i[k] for i in rows]

    with open(p, 'w') as fp:
        json.dump(logs, fp)
    return logs

# to_load = ['exchange_pair', 'exchange probas']
# for i in range(8):
#     to_load.append(f'replica_{i}_learning_rate')

to_load = ['exchange_loss_1', 'exchange_loss_2']

logs = load_logs("dzvinn/deep-tempering/3jzdp7qg", to_load)
print(len(logs['exchange_loss_1']))
# history = {int(k.split('_')[1]): {'learning_rate': logs[k]} for k in logs if k.startswith('replica')}
# history['exchange_pair'] = logs['exchange_pair']
# history['swap'] = [round(p, 0) for p in logs['exchange probas']]
# labels = assign_up_down_labels(history, [0.01, 0.015], 8, 'learning_rate')
# print(labels)
#
# ratios = calc_up_down_ratio(history, labels, [0.01, 0.015], 8, 'learning_rate')
# for k in sorted(ratios.keys()):
#     print(f'{k} : {ratios[k]}')
# plot_up_down_ratios(ratios)
