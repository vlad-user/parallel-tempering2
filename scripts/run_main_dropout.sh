rs=(42 23 56 78 01)

#python main.py --exp_name test_lenet_dropout_swap --exchange_type swap --hp_to_swap dropout_rate --epochs 500  --model_name lenet5 --model_builder lenet5_cifar10_builder  --lr_min 0.015 --lr_max 0.015 --dropout_rate_min 0.01 --dropout_rate_max 0.25 --n_replicas 8 --batch_size 128 --train_data_size 45000 --use_ensemble_model --random_seed $i --notes clean_experiment

for i in "${rs[@]}"
do
	python main.py --exp_name test_lenet_dropout_no_swap --exchange_type no_swap --hp_to_swap dropout_rate --epochs 500  --model_name lenet5 --model_builder lenet5_cifar10_builder  --lr_min 0.015 --lr_max 0.015 --dropout_rate_min 0.01 --dropout_rate_max 0.25 --n_replicas 8 --batch_size 128 --train_data_size 45000 --use_ensemble_model --notes clean_experiment
done

for i in "${rs[@]}"
do
	python main.py --exp_name test_lenet_dropout_swap --exchange_type swap --hp_to_swap dropout_rate --epochs 500  --model_name lenet5 --model_builder lenet5_cifar10_builder  --lr_min 0.015 --lr_max 0.015 --dropout_rate_min 0.01 --dropout_rate_max 0.25 --n_replicas 8 --batch_size 128 --train_data_size 45000 --use_ensemble_model --random_seed $i --notes clean_experiment

done

for i in "${rs[@]}"
do
	python main.py --exp_name test_lenet_dropout_swap_sort --exchange_type swap_sort --hp_to_swap dropout_rate --epochs 500  --model_name lenet5 --model_builder lenet5_cifar10_builder  --lr_min 0.015 --lr_max 0.015 --dropout_rate_min 0.01 --dropout_rate_max 0.25 --n_replicas 8 --batch_size 128 --train_data_size 45000 --use_ensemble_model --random_seed $i --notes  test_C_effect  --proba_coeff_C  3.

done

for i in "${rs[@]}"
do
	python main.py --exp_name test_lenet_dropout_swap_sort --exchange_type swap_sort --hp_to_swap dropout_rate --epochs 500  --model_name lenet5 --model_builder lenet5_cifar10_builder  --lr_min 0.015 --lr_max 0.015 --dropout_rate_min 0.01 --dropout_rate_max 0.25 --n_replicas 8 --batch_size 128 --train_data_size 45000 --use_ensemble_model --random_seed $i --notes  test_C_effect  --proba_coeff_C  4.

done

for i in "${rs[@]}"
do
	python main.py --exp_name test_lenet_dropout_swap_sort --exchange_type swap_sort --hp_to_swap dropout_rate --epochs 500  --model_name lenet5 --model_builder lenet5_cifar10_builder  --lr_min 0.015 --lr_max 0.015 --dropout_rate_min 0.01 --dropout_rate_max 0.25 --n_replicas 8 --batch_size 128 --train_data_size 45000 --use_ensemble_model --random_seed $i --notes  test_C_effect  --proba_coeff_C  6.46

done

for i in "${rs[@]}"
do
	python main.py --exp_name test_lenet_dropout_swap_sort_adj --exchange_type swap_sort_adj --hp_to_swap dropout_rate --epochs 500  --model_name lenet5 --model_builder lenet5_cifar10_builder  --lr_min 0.015 --lr_max 0.015 --dropout_rate_min 0.01 --dropout_rate_max 0.25 --n_replicas 8 --batch_size 128 --train_data_size 45000 --use_ensemble_model --random_seed $i --notes clean_experiment

done


