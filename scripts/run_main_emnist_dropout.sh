rs=(42 23 56 78 01)

for i in "${rs[@]}"
do
	python ../main.py --exp_name test_lenet_emnist_dropout_no_swap --exchange_type no_swap --hp_to_swap dropout_rate --epochs 300  --model_name lenet5 --model_builder lenet5_emnist_builder  --lr_min 0.04 --lr_max 0.04 --dropout_rate_min 0.01 --dropout_rate_max 0.15 --n_replicas 8 --batch_size 128 --train_data_size 112320 --use_ensemble_model --notes clean_experiment --dataset_name emnist --random_seed $i
done

for i in "${rs[@]}"
do
	python ../main.py --exp_name test_lenet_emnist_dropout_swap --exchange_type swap --hp_to_swap dropout_rate --epochs 300  --model_name lenet5 --model_builder lenet5_emnist_builder  --lr_min 0.04 --lr_max 0.04 --dropout_rate_min 0.01 --dropout_rate_max 0.15 --n_replicas 8 --batch_size 128 --train_data_size 112320 --use_ensemble_model --random_seed $i --notes clean_experiment --dataset_name emnist  --proba_coeff_C 3.

done

for i in "${rs[@]}"
do
	python ../main.py --exp_name test_lenet_emnist_dropout_swap_sort --exchange_type swap_sort --hp_to_swap dropout_rate --epochs 300  --model_name lenet5 --model_builder lenet5_emnist_builder  --lr_min 0.04 --lr_max 0.04 --dropout_rate_min 0.01 --dropout_rate_max 0.15 --n_replicas 8 --batch_size 128 --train_data_size 112320 --use_ensemble_model --random_seed $i --notes clean_experiment --dataset_name emnist  --proba_coeff_C 2.

done


for i in "${rs[@]}"
do
	python ../main.py --exp_name test_lenet_emnist_dropout_swap_sort --exchange_type swap_sort --hp_to_swap dropout_rate --epochs 300  --model_name lenet5 --model_builder lenet5_emnist_builder  --lr_min 0.04 --lr_max 0.04 --dropout_rate_min 0.01 --dropout_rate_max 0.15 --n_replicas 8 --batch_size 128 --train_data_size 112320 --use_ensemble_model --random_seed $i --notes clean_experiment --dataset_name emnist  --proba_coeff_C 3.

done


for i in "${rs[@]}"
do
	python ../main.py --exp_name test_lenet_emnist_dropout_swap_sort --exchange_type swap_sort --hp_to_swap dropout_rate --epochs 300  --model_name lenet5 --model_builder lenet5_emnist_builder  --lr_min 0.04 --lr_max 0.04 --dropout_rate_min 0.01 --dropout_rate_max 0.15 --n_replicas 8 --batch_size 128 --train_data_size 112320 --use_ensemble_model --random_seed $i --notes clean_experiment --dataset_name emnist  --proba_coeff_C 4.

done

for i in "${rs[@]}"
do
	python ../main.py --exp_name test_lenet_emnist_dropout_swap_sort_adj --exchange_type swap_sort_adj --hp_to_swap dropout_rate --epochs 300  --model_name lenet5 --model_builder lenet5_emnist_builder  --lr_min 0.04 --lr_max 0.04 --dropout_rate_min 0.01 --dropout_rate_max 0.15 --n_replicas 8 --batch_size 128 --train_data_size 112320 --use_ensemble_model --random_seed $i --notes clean_experiment --dataset_name emnist  --proba_coeff_C 3.

done

