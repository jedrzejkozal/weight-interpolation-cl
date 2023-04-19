import itertools
import mlflow
import subprocess

from main import *


def main():
    # all values for arguements defeined in model
    hyperparameters_invervals = {
        'ewc_on': {
            'lr': (0.1, 0.01, 0.001),
            'e_lambda': (0.1, 0.01, 0.001),
            'gamma': (0.0,),
        },
    }

    lecun_fix()
    parser, args = parse_known_args()
    mod = importlib.import_module('models.' + args.model)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--device', type=str, default='cuda:0')
    if hasattr(mod, 'Buffer'):
        parser.add_argument('--buffer_size', type=int, required=True,
                            help='The size of the memory buffer.')
    parser.add_argument('--n_epochs', type=int,
                        help='Batch size.')

    args = parser.parse_args()
    # args = parse_args()

    assert args.model in hyperparameters_invervals.keys(), f"model {args.model} has undefined hyperparameters search intervals"
    search_space = hyperparameters_invervals[args.model]
    assert args.experiment_name != 'Default'

    args.seed = 3141592
    run_name = f'{args.model} hyperparameters'
    args.run_name = run_name

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        id = mlflow.create_experiment(args.experiment_name)
        experiment = client.get_experiment(id)
    experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as active_run:
        parrent_run_id = active_run.info.run_id

    grid_search(args, search_space, parrent_run_id)

    n_repeats = 5
    args = select_best_paramters(args, client, experiment_id, parrent_run_id)
    # args.forgetting_stopping_threshold = 1.0

    with mlflow.start_run(experiment_id=experiment_id, run_name=f'{run_name} final') as final_run:
        args.parent_run_id = final_run.info.run_id
    for repeat in range(n_repeats):
        args.run_name = f'{args.model} final run {repeat}'
        args.seed = 43 + repeat
        run_training(args)


def grid_search(args, search_space: dict, parent_run_id: str):
    names = []
    intervals = []
    for hyperparam_name, interval in search_space.items():
        names.append(hyperparam_name)
        intervals.append(interval)

    grid = itertools.product(*intervals)
    args.parent_run_id = parent_run_id

    for hyperparm_values in grid:
        run_name = f'{args.model}'
        for name, value in zip(names, hyperparm_values):
            setattr(args, name, value)
            run_name += f' {name}={value}'
        args.run_name = run_name

        run_training(args)


def run_training(args):
    command = f'python main.py'
    for name, value in vars(args).items():
        if name == 'load_best_args':
            continue
        elif name == 'run_name':
            command += f' --run_name="{value}"'
        else:
            command += f' --{name}={value}'
    print('running commmand: ', command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()


def select_best_paramters(args, client, experiment_id, parrent_run_id):
    runs_list = select_runs_with_parent(client, parrent_run_id, experiment_id)
    best_run = select_best(runs_list)
    best_parameters = best_run.data.params

    arg_names = list(vars(args).keys())
    for name in arg_names:
        if name == 'load_best_args':
            continue
        value = best_parameters[name]
        setattr(args, name, value)

    print('\nbest args')
    for name, value in vars(args).items():
        print(f'\t{name}: {value}, type = {type(value)}')

    return args


def select_runs_with_parent(client, parrent_run_id, experiemnt_id):
    runs = mlflow.search_runs(experiment_ids=[experiemnt_id])
    run_ids = runs['run_id']

    selected_runs = []
    for run_id in run_ids:
        run = mlflow.get_run(run_id)
        run_data = run.data
        if 'mlflow.parentRunId' not in run_data.tags:
            continue
        parent_run = run_data.tags['mlflow.parentRunId']
        if parent_run != parrent_run_id:
            continue
        selected_runs.append(run)

    return selected_runs


def select_best(runs_list):
    best_run = None
    best_avrg_acc = 0.0
    for run in runs_list:
        run_metrics = run.data.metrics
        avrg_acc = run_metrics['mean_acc_class_il']
        if avrg_acc > best_avrg_acc:
            best_avrg_acc = avrg_acc
            best_run = run

    return best_run


if __name__ == '__main__':
    main()
