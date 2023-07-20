#! /bin/bash



# hyperparameters tuning
for N_TASKS in 20 10 5
do
    python hyperparameters.py --model="sgd" --dataset="seq-cifar100" --n_tasks=$N_TASKS --experiment_name="seq-cifar100" --n_epochs=50 --batch_size=32 --device="cuda:1"
    python hyperparameters.py --model="mir" --dataset="seq-cifar100" --n_tasks=5 --experiment_name="seq-cifar100" --buffer_size=500 --n_epochs=50 --batch_size=32 --minibatch_size=64 --device="cuda:1"
done