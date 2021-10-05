import numpy as np
import os
import json
import wandb

api = wandb.Api()
exp_ids = """
2zcnog9f
1s87tl66
2m202qv2
2sg5adid
166ukmez
""".split('\n')
exp_ids.reverse()
exp_ids.pop(0)


N_REPLICAS = 8

def load_logs(run_path, metrics_to_load):
    p = os.path.join('logs', run_path.split('/')[-1] + '.json')
    # print(p)
    if os.path.exists(p):
        with open(p, 'r') as fp:
            logs = json.load(fp)
            if all([k in logs.keys() for k in metrics_to_load]):
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

def calc_test_error_at_the_end(logs):
    test_errors = np.array([[[(1 - val) * 100 for val in l[f'val_acc_{i}']] for i in range(N_REPLICAS)] for l in logs])
    return np.mean(np.min(test_errors[:,:, -1], axis=1)), np.std(np.min(test_errors[:,:, -1], axis=1))

def calc_lowest_test_error(logs):
    all_best_val_accs = np.array([(1 - l['best val acc, # of replica, step'][0][0]) * 100. for l in logs]).reshape(-1, len(logs))
    print(all_best_val_accs)
    return np.mean(all_best_val_accs, axis=1)[0], np.std(all_best_val_accs, axis=1)[0]


def calc_accpt_ratio(logs):
    if len(logs[0]['swaped']) > 0:
        accpt_ratios = np.array([sum(l['swaped']) / len(l['swaped']) for l in logs]).reshape(-1, len(logs))
        return np.mean(accpt_ratios, axis=1)[0], np.std(accpt_ratios, axis=1)[0]
    else:
        return 0, 0

def calc_num_stuck_replicas(logs):
    if len(logs[0]['swaped']) > 0:
        accpt_ratios = np.array([sum(l['swaped']) / len(l['swaped']) for l in logs]).reshape(-1, len(logs))
        return np.mean(accpt_ratios, axis=1)[0], np.std(accpt_ratios, axis=1)[0]
    else:
        return 0, 0


def calc_avg_num_visited_temp(logs):
    if len(logs[0]['swaped']) > 0:

        avg_num = np.array([l['avg_num_temp_repl_visited'][0] for l in logs]).reshape(-1, len(logs))
        return np.mean(avg_num, axis=1)[0], np.std(avg_num, axis=1)[0]
    else:
        return 0, 0

def calc_frac_repl_all_temp(logs):
    if len(logs[0]['swaped']) > 0:
        frac = np.array([l['frac_of_repl_visited_all_temp'][0] for l in logs]).reshape(-1, len(logs))
        return np.mean(frac, axis=1)[0], np.std(frac, axis=1)[0]
    else:
        return 0, 0


metrics = {'test error at the end': calc_test_error_at_the_end,
           'lowest test error': calc_lowest_test_error,
           'acceptance ratio': calc_accpt_ratio,
           'avg num of visited temp': calc_avg_num_visited_temp,
           'frac of replicas that visited all temp': calc_frac_repl_all_temp}


to_load = ['best val acc, # of replica, step',  'avg_num_temp_repl_visited', 'frac_of_repl_visited_all_temp', 'swaped']
for r in range(N_REPLICAS):
    to_load.append(f'val_acc_{r}')


for i in [0]:
    exp_group = exp_ids[i:i+5]
    print(exp_group)
    logs = [load_logs(f'dzvinn/deep-tempering/{exp_id}', to_load) for exp_id in exp_group]
    for m in metrics:
        value = metrics[m](logs)
        print(f'{m}: {np.round(value[0], 3)} +- {np.round(value[1], 3)}')