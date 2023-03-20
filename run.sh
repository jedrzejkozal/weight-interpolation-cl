#! /bin/bash

python main.py --model="clewi" --dataset="seq-cifar100" --buffer_size=500 --seed=43 --lr=0.1 --run_name="clewi run 1"
python main.py --model="clewi" --dataset="seq-cifar100" --buffer_size=500 --seed=44 --lr=0.1 --run_name="clewi run 2"
python main.py --model="clewi" --dataset="seq-cifar100" --buffer_size=500 --seed=45 --lr=0.1 --run_name="clewi run 3"
python main.py --model="clewi" --dataset="seq-cifar100" --buffer_size=500 --seed=46 --lr=0.1 --run_name="clewi run 4"
python main.py --model="clewi" --dataset="seq-cifar100" --buffer_size=500 --seed=47 --lr=0.1 --run_name="clewi run 5"

python main.py --model="er" --load_best_args --dataset="seq-cifar100" --buffer_size=500 --seed=43 --run_name="er run 1"
python main.py --model="er" --load_best_args --dataset="seq-cifar100" --buffer_size=500 --seed=44 --run_name="er run 2"
python main.py --model="er" --load_best_args --dataset="seq-cifar100" --buffer_size=500 --seed=45 --run_name="er run 3"
python main.py --model="er" --load_best_args --dataset="seq-cifar100" --buffer_size=500 --seed=46 --run_name="er run 4"
python main.py --model="er" --load_best_args --dataset="seq-cifar100" --buffer_size=500 --seed=47 --run_name="er run 5"