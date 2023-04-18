import pathlib
import numpy as np
import mlflow
import tabulate


def main():
    standard_benchmarks()
    varing_n_tasks()


def standard_benchmarks():
    runs_standard_benchmarks = {
        'cifar10': {
            'Joint': None,
            'Finetuning': None,
            'EWC': None,
            'SI': None,
            'iCARL': None,
            'ER': None,
            'aGEM': None,
            'BIC': None,
            'ER-ACE': None,
            'GDumb': None,
            'MER': None,
            'MIR': None,
            'DER': None,
            'xDER': None,
            'CLeWI': None,
        },
        'cifar100': {
            'Joint': None,
            'Finetuning': None,
            'EWC': None,
            'SI': None,
            'iCARL': ['94de55f367924c248a45c5987df33524', 'c8e0f027b51c49648f69f0eb38509462', '610e0073957d49548b68791dd9246919', '7a08a1afe0c741f6a5e893681aa9f8a2', 'de486b95a6c74bfda0a9dd1dd1ec246d'],
            'ER': ['b493ab8e54004750896dbf9cebf0b609', '1951a520b7974b7c9a6d2e26f70f1d3c', '997ecc0f525f4071a74f0b216e9f39c8', 'a8513f0c1db74676af3b9454f49e7a2f', 'e011a0cbc9854328b248df835ae5d178'],
            'aGEM': None,
            'BIC': ['3baf04da53404ca4bbcc0506c90545bb', '5b0b5502cc82427f80df476408ce7da5', '12d168011906473aa728600e672f91d8', '1e686f8e6e4e4c96b113adfe1547e01c'],
            'ER-ACE': ['a7e438cff0b64590a1b51e421bc200fc', '1b04f1d16d584333826c3c29d277633b', '459fb22963c04042800b2492307c208d', '9f2060e2ea934c109d5447895fddbfcc'],
            'GDumb': ['ff544949cdda41dcb17666474f40db51', '330650c362984bd2abf34d6fbad93366', 'c3e0e9ddc19f4e1c8368411a54b31773', 'd11cb963b54c4a64ba58c50f86651eb6'],
            'MER': None,
            'MIR': None,
            'DER': ['cd06015c7db24f308176ed5a2bb632d9', '6946c90170914e319cd522c2563995a8', '22fb4bca74c0441d8fc7314da250b4ba'],
            'xDER': ['f4510037fdbf409d9a39c633d9961f33', '6279780737c64700af2dd780637300f4', '29f2ff8dba584728ac379c76e54341ab'],
            'CLeWI': None,
        },
        'tiny-imagenet': {
            'Joint': None,
            'Finetuning': None,
            'EWC': None,
            'SI': None,
            'iCARL': None,
            'ER': None,
            'aGEM': None,
            'BIC': None,
            'ER-ACE': None,
            'GDumb': None,
            'MER': None,
            'MIR': None,
            'DER': None,
            'xDER': None,
            'CLeWI': None,
        }
    }

    assert runs_standard_benchmarks['cifar10'].keys() == runs_standard_benchmarks['cifar100'].keys() == runs_standard_benchmarks['tiny-imagenet'].keys()
    algorithms = list(runs_standard_benchmarks['cifar10'].keys())

    mlruns_path = '///home/jkozal/Documents/PWr/interpolation/weight-interpolation-cl/mlruns/'
    # mlruns_path = '///home/jedrzejkozal/Documents/adversarial-computer-security/mlruns/'
    client = mlflow.tracking.MlflowClient(mlruns_path)

    dataset_experiments = {
        'cifar10': None,
        'cifar100': '675415310966171557',
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


def varing_n_tasks():
    runs_n_tasks = {
        '5 tasks': {
            'ER': None,
            'DER': None,
            'CLeWI': None,
        },
        '10 tasks': {
            'ER': None,
            'DER': None,
            'CLeWI': None,
        },
        '20 tasks': {
            'ER': None,
            'DER': None,
            'CLeWI': None,
        }
    }

    assert runs_n_tasks['5 tasks'].keys() == runs_n_tasks['10 tasks'].keys() == runs_n_tasks['20 tasks'].keys()
    algorithms = list(runs_n_tasks['5 tasks'].keys())

    mlruns_path = '///home/jkozal/Documents/PWr/interpolation/weight-interpolation-cl/mlruns/'
    # mlruns_path = '///home/jedrzejkozal/Documents/adversarial-computer-security/mlruns/'
    client = mlflow.tracking.MlflowClient(mlruns_path)

    table = list()
    for algorithm_name in algorithms:
        row = list()
        row.append(algorithm_name)
        for n_tasks in ('5 tasks', '10 tasks', '20 tasks'):
            run_ids = runs_n_tasks[n_tasks][algorithm_name]
            n_tasks = int(n_tasks.split()[0])
            acc, _ = calc_average_metrics(run_ids, client, '675415310966171557', n_tasks)
            row.append(acc)
        table.append(row)

    tab_latex = tabulate.tabulate(table, tablefmt="latex", headers=['method', '5 tasks', '10 tasks', '20 tasks',])
    tab_latex = tab_latex.replace('\\textbackslash{}', '\\')
    tab_latex = tab_latex.replace('\\{', '{')
    tab_latex = tab_latex.replace('\\}', '}')
    print(tab_latex)
    print("\n\n")


def calc_average_metrics(dataset_run_ids, client, experiment_id, n_tasks=20):
    if dataset_run_ids == None:
        return '-', '-'

    acc_all = []
    fm_all = []
    for run_id in dataset_run_ids:
        acc = get_metrics(run_id, client)
        acc_all.append(acc)
        fm = calc_forgetting_measure(run_id, client, experiment_id=experiment_id, num_tasks=n_tasks)  # TODO fix logging num_tasks in experiments
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
