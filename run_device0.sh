#! /bin/bash

#! /bin/bash

# 20 tasks
for SEED in 43 44 45 46 47
do 
    # no best args
    # python main.py --model="ewc_on" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=20 --run_name="ewc_on seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    # no best args
    # python main.py --model="si" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=20 --run_name="ewc_on seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    python main.py --model="icarl" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=20 --buffer_size=500 --run_name="icalr seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    python main.py --model="er" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=20 --buffer_size=500 --run_name="er seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    # no best args
    # python main.py --model="agem" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=20 --run_name="agem seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    python main.py --model="bic" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=20 --buffer_size=500 --run_name="bic seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    python main.py --model="er_ace" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=20 --buffer_size=500 --run_name="er_ace seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    python main.py --model="gdumb" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=20 --buffer_size=500 --run_name="gdumb seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
done


# 10 tasks
for SEED in 43 44 45 46 47
do 
    # no best args
    # python main.py --model="ewc_on" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=10 --run_name="ewc_on seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    # no best args
    # python main.py --model="si" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=10 --run_name="ewc_on seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    python main.py --model="icarl" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=10 --buffer_size=500 --run_name="icalr seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    python main.py --model="er" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=10 --buffer_size=500 --run_name="er seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    # no best args
    # python main.py --model="agem" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=10 --run_name="agem seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    python main.py --model="bic" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=10 --buffer_size=500 --run_name="bic seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    python main.py --model="er_ace" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=10 --buffer_size=500 --run_name="er_ace seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    python main.py --model="gdumb" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=10 --buffer_size=500 --run_name="gdumb seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
done

# 5 tasks
for SEED in 43 44 45 46 47
do 
    # no best args
    # python main.py --model="ewc_on" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=5 --run_name="ewc_on seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    # no best args
    # python main.py --model="si" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=5 --run_name="ewc_on seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    python main.py --model="icarl" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=5 --buffer_size=500 --run_name="icalr seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    python main.py --model="er" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=5 --buffer_size=500 --run_name="er seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    # no best args
    # python main.py --model="agem" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=5 --run_name="agem seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    python main.py --model="bic" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=5 --buffer_size=500 --run_name="bic seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    python main.py --model="er_ace" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=5 --buffer_size=500 --run_name="er_ace seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
    python main.py --model="gdumb" --load_best_args --dataset="seq-cifar100" --seed=$SEED --n_tasks=5 --buffer_size=500 --run_name="gdumb seed $SEED 20 tasks" --experiment_name="seq-cifar100" --device="cuda:0"
done