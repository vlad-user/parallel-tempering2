#rs=(0.0005 0.001 0.004 0.008 0.01 0.05)
rs=(42)

#python ../main.py --exp_name test_resnet_lr_original_hps_rs_42 --exchange_type no_swap --hp_to_swap learning_rate --epochs 250  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.1 --lr_max 0.1 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 5 --batch_size 128 --train_data_size 45000  --random_seed 42 --notes  see_if_val_loss_increase_with_keras_model  --proba_coeff_C  2.5
#python ../main.py --exp_name test_resnet_lr_original_hps_rs_50 --exchange_type no_swap --hp_to_swap learning_rate --epochs 250  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.1 --lr_max 0.1 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 5 --batch_size 128 --train_data_size 45000  --random_seed 50 --notes  see_if_val_loss_increase_with_keras_model  --proba_coeff_C  2.5
#python ../main.py --exp_name test_resnet_lr_original_hps_rs_60 --exchange_type no_swap --hp_to_swap learning_rate --epochs 250  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.1 --lr_max 0.1 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 5 --batch_size 128 --train_data_size 45000  --random_seed 60 --notes  see_if_val_loss_increase_with_keras_model  --proba_coeff_C  2.5



for i in "${rs[@]}"
do

#  	python ../main.py --exp_name test_resnet_lr_no_swap --exchange_type no_swap --hp_to_swap learning_rate --epochs 300  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.0007 --lr_max 0.01 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 12 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes not_equidistant_temp    --proba_coeff_C  1. --use_ensemble_model
#    python ../main.py --exp_name test_resnet_lr_swap_sort --exchange_type swap_sort --hp_to_swap learning_rate --epochs 350  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.0007 --lr_max 0.007 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 12 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes not_equidistant_temp    --proba_coeff_C  1. --use_ensemble_model
                python ../main.py --exp_name test_resnet_lr_swap_sort --exchange_type swap_sort --hp_to_swap learning_rate --epochs 350  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.0007 --lr_max 0.007 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 12 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes not_equidistant_temp    --proba_coeff_C  0.5 --use_ensemble_model

        python ../main.py --exp_name test_resnet_lr_swap_sort --exchange_type swap_sort --hp_to_swap learning_rate --epochs 350  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.0007 --lr_max 0.007 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 12 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes not_equidistant_temp    --proba_coeff_C  0.3 --use_ensemble_model

#        python ../main.py --exp_name test_resnet_lr_swap_sort_adj --exchange_type swap_sort_adj --hp_to_swap learning_rate --epochs 300  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.0007 --lr_max 0.01 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 12 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes not_equidistant_temp    --proba_coeff_C  1. --use_ensemble_model

#    python ../main.py --exp_name test_resnet_lr_swap_sort --exchange_type swap_sort --hp_to_swap learning_rate --epochs 300  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.0007 --lr_max 0.01 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 12 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes playing_with_num_replicas_and_lr_bounds    --proba_coeff_C  1. --use_ensemble_model



#	python ../main.py --exp_name test_resnet_lr_no_swap --exchange_type no_swap --hp_to_swap learning_rate --epochs 300  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.001 --lr_max 0.01 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 8 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes playing_with_num_replicas_and_lr_bounds    --proba_coeff_C  1. --use_ensemble_model
#
#  python ../main.py --exp_name test_resnet_lr_swap_sort --exchange_type swap_sort --hp_to_swap learning_rate --epochs 300  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.001 --lr_max 0.01 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 8 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes playing_with_num_replicas_and_lr_bounds    --proba_coeff_C  1. --use_ensemble_model
#
#  python ../main.py --exp_name test_resnet_lr_no_swap --exchange_type no_swap --hp_to_swap learning_rate --epochs 300  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.001 --lr_max 0.01 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 12 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes playing_with_num_replicas_and_lr_bounds    --proba_coeff_C  1. --use_ensemble_model
#
#  python ../main.py --exp_name test_resnet_lr_swap_sort --exchange_type swap_sort --hp_to_swap learning_rate --epochs 300  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.001 --lr_max 0.01 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 12 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes playing_with_num_replicas_and_lr_bounds    --proba_coeff_C  1. --use_ensemble_model
#
#
#  python ../main.py --exp_name test_resnet_lr_no_swap --exchange_type no_swap --hp_to_swap learning_rate --epochs 300  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.001 --lr_max 0.05 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 12 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes playing_with_num_replicas_and_lr_bounds    --proba_coeff_C  1. --use_ensemble_model
#
#  python ../main.py --exp_name test_resnet_lr_swap_sort --exchange_type swap_sort --hp_to_swap learning_rate --epochs 300  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.001 --lr_max 0.05 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 12 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes playing_with_num_replicas_and_lr_bounds    --proba_coeff_C  1. --use_ensemble_model
#
#
#
#  python ../main.py --exp_name test_resnet_lr_swap_sort_adj --exchange_type swap_sort_adj --hp_to_swap learning_rate --epochs 300  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.001 --lr_max 0.01 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 8 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes playing_with_num_replicas_and_lr_bounds    --proba_coeff_C  1. --use_ensemble_model



#  python ../main.py --exp_name test_resnet_lr_swap_sort --exchange_type swap_sort --hp_to_swap learning_rate --epochs 250  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.001 --lr_max 0.05 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 8 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes  make_sure_lr_bounds_are_ok  --proba_coeff_C  3. --use_ensemble_model
#
#  python ../main.py --exp_name test_resnet_lr_swap_sort_adj --exchange_type swap_sort_adj --hp_to_swap learning_rate --epochs 250  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.001 --lr_max 0.05 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 8 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes  make_sure_lr_bounds_are_ok  --proba_coeff_C  1. --use_ensemble_model
#
#  python ../main.py --exp_name test_resnet_lr_no_swap --exchange_type no_swap --hp_to_swap learning_rate --epochs 250  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.0005 --lr_max 0.001 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 12 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes  more_replicas  --proba_coeff_C  1. --use_ensemble_model
#
#  python ../main.py --exp_name test_resnet_lr_swap_sort --exchange_type swap_sort --hp_to_swap learning_rate --epochs 250  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.0005 --lr_max 0.001 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 12 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes  more_replicas  --proba_coeff_C  1. --use_ensemble_model
#
#  python ../main.py --exp_name test_resnet_lr_swap_sort_adj --exchange_type swap_sort_adj --hp_to_swap learning_rate --epochs 250  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.001 --lr_max 0.05 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 12 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes  make_sure_lr_bounds_are_ok  --proba_coeff_C  1. --use_ensemble_model


done