import mlflow
import pathlib
import numpy as np
import matplotlib.pyplot as plt


def main():
    width_runs = {
        'clewi': {
            '1': ['cf7c6b8b65b04450a468d2f2fa0e9ac9', '5b0b0cec489644b789105d0852223c02', '189b17529d5d49dfb549d1c3d3256306'],
            '2': ['1772099fdb5c4f8289313920e48ff3d8', '4a5ebec62798441db255cd8d842d824a', 'e36fbbb1cca046a181f219c41713e125'],
            '4': None,
            '8': None,
        },
        'er': {
            '1': ['46fda71735f64c8da98d050960dd3cd9', '7835e0307ca143689620fced42f62c9d', '2239aa1376dd4467852af4347adf4811'],
            '2': ['c271df8e94f344a6a6a851d9ba22f889', '5488420cbd564042b326bda3b7ceae2f', '026d42bad63344fdb9815c9d6c06abee'],
            '4': None,
            '8': None,
        },
        'derpp': {
            '1': ['ae8b5183dcf2473098e4d6a2c040f8d7', 'a8002e911d8343f99aec7a59d67c34bb', '186e1034cd914a658c83d2c1c076df27'],
            '2': None,
            '4': None,
            '8': None,
        },
    }

    algorithms = list(width_runs.keys())

    mlruns_path = '///home/jkozal/Documents/PWr/interpolation/weight-interpolation-cl/mlruns/'
    # mlruns_path = '///home/jedrzejkozal/Documents/adversarial-computer-security/mlruns/'
    client = mlflow.tracking.MlflowClient(mlruns_path)

    widths = [1, 2, 4, 8]
    for algorithm_name in algorithms:
        width_acc = list()
        for width in width_runs[algorithm_name].keys():
            run_ids = width_runs[algorithm_name][width]
            if width == '1':
                experiment_id = '675415310966171557'
            else:
                experiment_id = '915839014035161355'
            acc_avrg, acc_std, fm_avrg, fm_std, last_acc_avrg, last_acc_std = calc_average_metrics(run_ids, client, experiment_id)
            width_acc.append(acc_avrg)
        plt.plot(widths, width_acc, label=algorithm_name)
    plt.xticks([1, 2, 4, 8])
    plt.xlabel('network width')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def calc_average_metrics(dataset_run_ids, client, experiment_id, n_tasks=10):
    if dataset_run_ids == None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    acc_all = []
    fm_all = []
    last_task_acc_all = []
    for run_id in dataset_run_ids:
        acc = get_metrics(run_id, client)
        acc_all.append(acc)
        fm = calc_forgetting_measure(run_id, client, experiment_id=experiment_id, num_tasks=n_tasks)  # TODO fix logging num_tasks in experiments
        fm_all.append(fm)
        last_task_acc = get_last_task_acc(run_id, client, experiment_id=experiment_id, num_tasks=n_tasks)
        last_task_acc_all.append(last_task_acc)

    avrg_acc, acc_std = reduction(acc_all)
    avrg_fm, fm_std = reduction(fm_all)
    avrg_last_acc, last_acc_std = reduction(last_task_acc_all)
    return avrg_acc, acc_std, avrg_fm, fm_std, avrg_last_acc, last_acc_std


def get_metrics(run_id, client):
    run = client.get_run(run_id)
    run_metrics = run.data.metrics
    acc = run_metrics['mean_acc_class_il']
    return acc


def reduction(metrics):
    metrics = np.array(metrics)
    avrg = metrics.mean()
    std = metrics.std()
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


def get_last_task_acc(run_id, client, experiment_id, num_tasks=None):
    run_path = pathlib.Path(f'mlruns/{experiment_id}/{run_id}/metrics/')
    if num_tasks is None:
        run = client.get_run(run_id)
        num_tasks = run.data.params['n_experiences']
        num_tasks = int(num_tasks)

    filepath = run_path / f'acc_class_il_task_{num_tasks-1}'
    with open(filepath, 'r') as f:
        for line in f.readlines():
            acc_str = line.split()[-2]
            last_acc = float(acc_str)

    return last_acc


if __name__ == '__main__':
    main()
