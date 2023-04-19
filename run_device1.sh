#! /bin/bash

# 20 tasks
for SEED in 43 44 45 46 47
do 
    # no best args
    # python main.py --model="mer" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=20 --buffer_size=500 --run_name="mer seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    # no best args
    # python main.py --model="mir" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=20 --buffer_size=500 --run_name="mir seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    python main.py --model="derpp" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=20 --buffer_size=500 --run_name="derpp seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    python main.py --model="xder" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=20 --buffer_size=500 --run_name="xder seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:1"

    python main.py --model="clewim" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=20 --lr=0.1 --run_name="clewi seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    # no best args
    # python main.py --model="clewi_er_ace" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=20 --lr=0.1 --run_name="clewi_er_ace seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    python main.py --model="clewi_derpp" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=20 --lr=0.1 --run_name="clewi_derpp seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    python main.py --model="clewi_xder" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=20 --lr=0.1 --run_name="clewi_xder seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
done


# 10 tasks
for SEED in 43 44 45 46 47
do 
    # no best args
    # python main.py --model="mer" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=10 --buffer_size=500 --run_name="mer seed $SEED 10 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    # no best args
    # python main.py --model="mir" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=10 --buffer_size=500 --run_name="mir seed $SEED 10 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    python main.py --model="derpp" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=10 --buffer_size=500 --run_name="derpp seed $SEED 10 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    python main.py --model="xder" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=10 --buffer_size=500 --run_name="xder seed $SEED 10 tasks" --experiment_name="seq-cifar100" --device="cuda:1"

    python main.py --model="clewim" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=10 --lr=0.1 --run_name="clewi seed $SEED 10 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    # no best args
    # python main.py --model="clewi_er_ace" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=10 --lr=0.1 --run_name="clewi_er_ace seed $SEED 10 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    python main.py --model="clewi_derpp" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=10 --lr=0.1 --run_name="clewi_derpp seed $SEED 10 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    python main.py --model="clewi_xder" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=10 --lr=0.1 --run_name="clewi_xder seed $SEED 10 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
done

# 5 tasks
for SEED in 43 44 45 46 47
do 
    # no best args
    # python main.py --model="mer" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=5 --buffer_size=500 --run_name="mer seed $SEED 5 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    # no best args
    # python main.py --model="mir" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=5 --buffer_size=500 --run_name="mir seed $SEED 5 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    python main.py --model="derpp" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=5 --buffer_size=500 --run_name="derpp seed $SEED 5 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    python main.py --model="xder" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=5 --buffer_size=500 --run_name="xder seed $SEED 5 tasks" --experiment_name="seq-cifar100" --device="cuda:1"

    python main.py --model="clewim" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=5 --lr=0.1 --run_name="clewi seed $SEED 5 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    # no best args
    # python main.py --model="clewi_er_ace" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=5 --lr=0.1 --run_name="clewi_er_ace seed $SEED 5 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    python main.py --model="clewi_derpp" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=5 --lr=0.1 --run_name="clewi_derpp seed $SEED 5 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
    python main.py --model="clewi_xder" --dataset="seq-cifar100" --buffer_size=500 --seed=$SEED --n_tasks=5 --lr=0.1 --run_name="clewi_xder seed $SEED 5 tasks" --experiment_name="seq-cifar100" --device="cuda:1"
done