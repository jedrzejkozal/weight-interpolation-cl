# Continual Learning with Weight Interpolation

This repo contains code for the CLeWI paper.

We use following libraries:
 - mammoth library for Continual Learning [link](https://github.com/aimagelab/mammoth)
 - REPAIR algorithm for weight interpolation code from github repo [link](https://github.com/KellerJordan/REPAIR)
 - for loss landscape visualization we modify code from loss-landscapes library [link](https://github.com/marcellodebernardi/loss-landscapes)

To run any experiments please create and activate conda env:

```
conda env create -f env.yml -y
conda activate interpolation
```

Run CLeWI on CIFAR100 with 10 tasks:

```
python main.py --model="clewi" --dataset="seq-cifar100" --n_tasks=10 --lr=0.1 --buffer_size=500 --n_epochs=50 --seed=42 --optim_wd=0.0 --optim_mom=0.0
```

All results are stored in mlflow in thie repository. You can run mlflow ui server locally:

```
mlflow ui
```

And then go to [http://127.0.0.1:5000/#/](http://127.0.0.1:5000/#/) in your brower to see all the results from the experiments we runned and exact hyperparameters used in each run.

## Citation policy

Please cite our work as

```
@misc{kozal2024continual,
      title={Continual Learning with Weight Interpolation}, 
      author={Jędrzej Kozal and Jan Wasilewski and Bartosz Krawczyk and Michał Woźniak},
      year={2024},
      eprint={2404.04002},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```