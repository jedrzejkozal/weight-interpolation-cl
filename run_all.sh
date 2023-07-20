#! /bin/bash

# training with best hyperparameters known (from previous experiments)
for N_TASKS in 20 10 5
do
    for SEED in 43 44 45 46 47
    do 
        # no best args
        # python main.py --model="ewc_on" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --run_name="ewc_on seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        # no best args
        # python main.py --model="si" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --run_name="ewc_on seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        python main.py --model="icarl" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="icalr seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        python main.py --model="er" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="er seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        # no best args
        # python main.py --model="agem" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --run_name="agem seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        python main.py --model="bic" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="bic seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        python main.py --model="er_ace" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="er_ace seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        python main.py --model="gdumb" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="gdumb seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        # no best args
        # python main.py --model="mer" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="mer seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        # no best args
        # python main.py --model="mir" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="mir seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        python main.py --model="derpp" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="derpp seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        python main.py --model="xder" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=500 --run_name="xder seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"

        python main.py --model="clewim" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=$N_TASKS --lr=0.1 --run_name="clewi seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        # no best args
        # python main.py --model="clewi_er_ace" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=$N_TASKS --lr=0.1 --run_name="clewi_er_ace seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        python main.py --model="clewi_derpp" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=$N_TASKS --lr=0.1 --run_name="clewi_derpp seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
        python main.py --model="clewi_xder" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=$N_TASKS --lr=0.1 --run_name="clewi_xder seed $SEED $N_TASKS tasks" --experiment_name="seq-cifar100"
    done
done


# hyperparameters tuning
for N_TASKS in 20 10 5
do
    python hyperparameters.py --model="joint" --dataset="seq-cifar100" --n_tasks=$N_TASKS --experiment_name="seq-cifar100" --n_epochs=50 --batch_size=32
    python hyperparameters.py --model="sgd" --dataset="seq-cifar100" --n_tasks=$N_TASKS --experiment_name="seq-cifar100" --n_epochs=50 --batch_size=32
    # python hyperparameters.py --model="mer" --dataset="seq-cifar100" --n_tasks=$N_TASKS --experiment_name="seq-cifar100" --buffer_size=500 --n_epochs=50 --batch_size=1
    python hyperparameters.py --model="mir" --dataset="seq-cifar100" --n_tasks=5 --experiment_name="seq-cifar100" --buffer_size=500 --n_epochs=50 --batch_size=32 --minibatch_size=64
done