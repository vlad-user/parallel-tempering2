#rs=(0.0005 0.001 0.004 0.008 0.01 0.05)
rs=(42 50)

python ../main.py --exp_name test_resnet_lr_original_hps --exchange_type no_swap --hp_to_swap learning_rate --epochs 200  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.1 --lr_max 0.1 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 5 --batch_size 128 --train_data_size 45000  --random_seed 42 --notes  using_original_training_procedure  --proba_coeff_C  2.5 --use_ensemble_model


#for i in "${rs[@]}"
#do
#
#	python ../main.py --exp_name test_resnet_lr_no_swap --exchange_type no_swap --hp_to_swap learning_rate --epochs 200  --model_name resnet20 --model_builder resnet20_v1_cifar10_builder  --lr_min 0.0005 --lr_max 0.05 --dropout_rate_min 0. --dropout_rate_max 0. --n_replicas 5 --batch_size 128 --train_data_size 45000  --random_seed ${i} --notes  make_sure_lr_bounds_are_ok  --proba_coeff_C  2.5 --use_ensemble_model
#
#
#done