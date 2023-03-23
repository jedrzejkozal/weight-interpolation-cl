import pathlib
import numpy as np
import mlflow
import tabulate


def main():
    runs_standard_benchmarks = {
        'cifar10': {
            'ER': None,
            'DER': None,
            'CLeWI': None,
        },
        'cifar100': {
            # 'Upperbound': [],
            # 'Naive': [],
            # 'EWC': [],
            # 'SI': [],
            # 'GDumb': [],
            # 'aGEM': [],
            'ER': ['9e7759b915ac43e2a98cccea86add477', '7c955b150f4544069119659b612f7983', '0ceb7ce1d8c44577be0f0b5477093e39', 'd5e9b8b104d84a25ab8973c922d0058b', 'ea94709333714cdfb8ea746502c19005'],
            'DER': ['ba0f4ff573ca42fb98555abfda6a59f4', 'b1aa4fe10bfb44cc9b1e2a3dfd190092', 'bb3b983a406c42e0ba8edb4131dfbbb0', '42e6d96ed61442c3ac5fffdbe743a14f', 'e179d9b6482245c1b5a4b9dc8aa616cc'],
            # 'MIR': [],
            'CLeWI': ['e4e60735d2a24e46af776e58b665c50c', 'fc20d261f94a434b973d90d8f2cd53f8', 'd832df41be7f42da85dce970320f75b8', 'f1ca76b4d9254a88923fdbed686b3ae8', '7468c0d7ff51437a986d2857678214b5'],
        },
        'tiny-imagenet': {
            'ER': ['df1d15d952604f8eae5c6d7052c186cd', '8d5d2a422be04bfb92ad67034fad9e0a', '2e9a2795bc1545a4a38595e8ac4b9780', 'bd15ceeef00a4659a8800f58bf854511',],
            'DER': None,
            'CLeWI': ['3d16de643e934656a19f2974777b537f', '5835832d970f4d53929a8e3afbd0b397', '67586f57d45b4e2e8c54afb0fa854171', '8eadfcea2fbb4f86bc2f13e4f14a11ec', '25d5279fca464c84b6429a4915c5dbdf'],
        }
    }

    assert runs_standard_benchmarks['cifar10'].keys() == runs_standard_benchmarks['cifar100'].keys() == runs_standard_benchmarks['tiny-imagenet'].keys()
    algorithms = list(runs_standard_benchmarks['cifar10'].keys())

    mlruns_path = '///home/jkozal/Documents/PWr/interpolation/weight-interpolation-cl/mlruns/'
    # mlruns_path = '///home/jedrzejkozal/Documents/adversarial-computer-security/mlruns/'
    client = mlflow.tracking.MlflowClient(mlruns_path)

    dataset_experiments = {
        'cifar10': None,
        'cifar100': '0',
        'tiny-imagenet': '757344672409704114'
    }

    table = list()
    for algorithm_name in algorithms:
        row = list()
        row.append(algorithm_name)
        for dataset in ('cifar10', 'cifar100', 'tiny-imagenet'):
            run_ids = runs_standard_benchmarks[dataset][algorithm_name]
            experiment_id = dataset_experiments[dataset]
            metrics = calc_average_metrics(run_ids, client, experiment_id)
            row.extend(metrics)
        table.append(row)

    tab_latex = tabulate.tabulate(table, tablefmt="latex", headers=['method', 'acc', 'FM', 'acc', 'FM', 'acc', 'FM',])
    tab_latex = tab_latex.replace('\\textbackslash{}', '\\')
    tab_latex = tab_latex.replace('\\{', '{')
    tab_latex = tab_latex.replace('\\}', '}')
    print(tab_latex)
    print("\n\n")


def calc_average_metrics(dataset_run_ids, client, experiment_id):
    if dataset_run_ids == None:
        return '-', '-'

    acc_all = []
    fm_all = []
    for run_id in dataset_run_ids:
        acc = get_metrics(run_id, client)
        acc_all.append(acc)
        fm = calc_forgetting_measure(run_id, client, experiment_id=experiment_id, num_tasks=20)  # TODO fix logging num_tasks in experiments
        fm_all.append(fm)

    avrg_acc, acc_std = rounded_reduction(acc_all)
    acc = f'{avrg_acc}±{acc_std}'
    avrg_fm, fm_std = rounded_reduction(fm_all)
    forgetting = f'{avrg_fm}±{fm_std}'
    return acc, forgetting


def get_metrics(run_id, client):
    run = client.get_run(run_id)
    run_metrics = run.data.metrics
    acc = run_metrics['mean_acc_class_il']
    return acc


def rounded_reduction(metrics, digits=4):
    metrics = np.array(metrics)
    avrg = metrics.mean()
    avrg = round(avrg, digits)
    std = metrics.std()
    std = round(std, digits)
    return avrg, std


def calc_forgetting_measure(run_id, client, experiment_id, num_tasks=None):
    run_path = pathlib.Path(f'mlruns/{experiment_id}/{run_id}/metrics/')
    if num_tasks is None:
        run = client.get_run(run_id)
        num_tasks = run.data.params['n_experiences']
        num_tasks = int(num_tasks)

    fm = 0.0

    for task_id in range(num_tasks):
        filepath = run_path / f'acc_class_il_task_{task_id}'
        task_accs = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                acc_str = line.split()[-2]
                acc = float(acc_str)
                task_accs.append(acc)

        fm += abs(task_accs[-1] - max(task_accs))

    fm = fm / num_tasks
    return fm


if __name__ == '__main__':
    main()
