#! /bin/bash

# python main.py --model="clewi" --dataset="seq-cifar100" --buffer_size=500 --seed=43 --n_tasks=20 --lr=0.1 --run_name="clewi run 1 20 tasks"
# python main.py --model="clewi" --dataset="seq-cifar100" --buffer_size=500 --seed=44 --n_tasks=20 --lr=0.1 --run_name="clewi run 2 20 tasks"
# python main.py --model="clewi" --dataset="seq-cifar100" --buffer_size=500 --seed=45 --n_tasks=20 --lr=0.1 --run_name="clewi run 3 20 tasks"
# python main.py --model="clewi" --dataset="seq-cifar100" --buffer_size=500 --seed=46 --n_tasks=20 --lr=0.1 --run_name="clewi run 4 20 tasks"
# python main.py --model="clewi" --dataset="seq-cifar100" --buffer_size=500 --seed=47 --n_tasks=20 --lr=0.1 --run_name="clewi run 5 20 tasks"

# python main.py --model="er" --load_best_args --dataset="seq-cifar100" --buffer_size=500 --seed=43 --n_tasks=20 --run_name="er run 1 20 tasks"
# python main.py --model="er" --load_best_args --dataset="seq-cifar100" --buffer_size=500 --seed=44 --n_tasks=20 --run_name="er run 2 20 tasks"
# python main.py --model="er" --load_best_args --dataset="seq-cifar100" --buffer_size=500 --seed=45 --n_tasks=20 --run_name="er run 3 20 tasks"
# python main.py --model="er" --load_best_args --dataset="seq-cifar100" --buffer_size=500 --seed=46 --n_tasks=20 --run_name="er run 4 20 tasks"
# python main.py --model="er" --load_best_args --dataset="seq-cifar100" --buffer_size=500 --seed=47 --n_tasks=20 --run_name="er run 5 20 tasks"

# python main.py --model="der" --load_best_args --dataset="seq-cifar100" --buffer_size=500 --seed=43 --n_tasks=20 --run_name="der run 1 20 tasks"
# python main.py --model="der" --load_best_args --dataset="seq-cifar100" --buffer_size=500 --seed=44 --n_tasks=20 --run_name="der run 2 20 tasks"
# python main.py --model="der" --load_best_args --dataset="seq-cifar100" --buffer_size=500 --seed=45 --n_tasks=20 --run_name="der run 3 20 tasks"
# python main.py --model="der" --load_best_args --dataset="seq-cifar100" --buffer_size=500 --seed=46 --n_tasks=20 --run_name="der run 4 20 tasks"
# python main.py --model="der" --load_best_args --dataset="seq-cifar100" --buffer_size=500 --seed=47 --n_tasks=20 --run_name="der run 5 20 tasks"

python main.py --model="clewi" --dataset="seq-tinyimg" --buffer_size=500 --seed=43 --n_tasks=20 --lr=0.1 --run_name="clewi run 1 20 tasks" --experiment_name="seq-tinyimagenet"
python main.py --model="clewi" --dataset="seq-tinyimg" --buffer_size=500 --seed=44 --n_tasks=20 --lr=0.1 --run_name="clewi run 2 20 tasks" --experiment_name="seq-tinyimagenet"
python main.py --model="clewi" --dataset="seq-tinyimg" --buffer_size=500 --seed=45 --n_tasks=20 --lr=0.1 --run_name="clewi run 3 20 tasks" --experiment_name="seq-tinyimagenet"
python main.py --model="clewi" --dataset="seq-tinyimg" --buffer_size=500 --seed=46 --n_tasks=20 --lr=0.1 --run_name="clewi run 4 20 tasks" --experiment_name="seq-tinyimagenet"
python main.py --model="clewi" --dataset="seq-tinyimg" --buffer_size=500 --seed=47 --n_tasks=20 --lr=0.1 --run_name="clewi run 5 20 tasks" --experiment_name="seq-tinyimagenet"

python main.py --model="er" --load_best_args --dataset="seq-tinyimg" --buffer_size=500 --seed=43 --n_tasks=20 --run_name="er run 1 20 tasks" --experiment_name="seq-tinyimagenet"
python main.py --model="er" --load_best_args --dataset="seq-tinyimg" --buffer_size=500 --seed=44 --n_tasks=20 --run_name="er run 2 20 tasks" --experiment_name="seq-tinyimagenet"
python main.py --model="er" --load_best_args --dataset="seq-tinyimg" --buffer_size=500 --seed=45 --n_tasks=20 --run_name="er run 3 20 tasks" --experiment_name="seq-tinyimagenet"
python main.py --model="er" --load_best_args --dataset="seq-tinyimg" --buffer_size=500 --seed=46 --n_tasks=20 --run_name="er run 4 20 tasks" --experiment_name="seq-tinyimagenet"
python main.py --model="er" --load_best_args --dataset="seq-tinyimg" --buffer_size=500 --seed=47 --n_tasks=20 --run_name="er run 5 20 tasks" --experiment_name="seq-tinyimagenet"

python main.py --model="der" --load_best_args --dataset="seq-tinyimg" --buffer_size=500 --seed=43 --n_tasks=20 --run_name="der run 1 20 tasks" --experiment_name="seq-tinyimagenet"
python main.py --model="der" --load_best_args --dataset="seq-tinyimg" --buffer_size=500 --seed=44 --n_tasks=20 --run_name="der run 2 20 tasks" --experiment_name="seq-tinyimagenet"
python main.py --model="der" --load_best_args --dataset="seq-tinyimg" --buffer_size=500 --seed=45 --n_tasks=20 --run_name="der run 3 20 tasks" --experiment_name="seq-tinyimagenet"
python main.py --model="der" --load_best_args --dataset="seq-tinyimg" --buffer_size=500 --seed=46 --n_tasks=20 --run_name="der run 4 20 tasks" --experiment_name="seq-tinyimagenet"
python main.py --model="der" --load_best_args --dataset="seq-tinyimg" --buffer_size=500 --seed=47 --n_tasks=20 --run_name="der run 5 20 tasks" --experiment_name="seq-tinyimagenet"
