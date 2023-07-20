#! /bin/bash

# hyperparameters tuning
for N_TASKS in 20 10 5
    python hyperparameters.py --model="joint" --dataset="seq-cifar100" --n_tasks=$N_TASKS --experiment_name="seq-cifar100" --n_epochs=50 --batch_size=32 --device="cuda:0"
done



