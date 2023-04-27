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
            'BIC': ['3baf04da53404ca4bbcc0506c90545bb', '5b0b5502cc82427f80df476408ce7da5', '12d168011906473aa728600e672f91d8', '1e686f8e6e4e4c96b113adfe1547e01c', '25c878dddec74378b069173f3caf503f'],
            'ER-ACE': ['a7e438cff0b64590a1b51e421bc200fc', '1b04f1d16d584333826c3c29d277633b', '459fb22963c04042800b2492307c208d', '9f2060e2ea934c109d5447895fddbfcc', 'a41d246aac464288bbcd990e602c276b'],
            'GDumb': ['ff544949cdda41dcb17666474f40db51', '330650c362984bd2abf34d6fbad93366', 'c3e0e9ddc19f4e1c8368411a54b31773', 'd11cb963b54c4a64ba58c50f86651eb6', 'f1389212c51b455eae5d57f8eeac01d0'],
            'MER': None,
            'MIR': None,
            'DER': ['cd06015c7db24f308176ed5a2bb632d9', '6946c90170914e319cd522c2563995a8', '22fb4bca74c0441d8fc7314da250b4ba', 'df57945b9e304e94b6eb76c61e9778ac', '3157516230564a7887a4a4164843e12d'],
            'xDER': ['f4510037fdbf409d9a39c633d9961f33', '6279780737c64700af2dd780637300f4', '29f2ff8dba584728ac379c76e54341ab', '95383a9572b94a0180fa62dfb4bb01b6', '27b6ab60af294ce3a40ac56a33ffbe1b'],
            'CLeWI': ['7157360c5e63454d9dd11a794f7a4319', '98284fb1575e4863a79b1e3a2b9d2ab6', '4cfed79122ab440aa81d5f4f1006b70a', 'bace4c4d02c24be2b923e5d4d8e84289'],
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
            'Joint': None,
            'Finetuning': None,
            'EWC': None,
            'SI': None,
            'iCARL': ['7dc675684dfa4f8cacf8aeb1cc09ffc1', 'c9ed7dd02f5649b3b9792c40576832c7', 'f0bda601517343aaaba222367d242233', 'c604e22e957f4ccf98d251babcabc60a', '77dd9b747c4d45aab6d8011540eb4fad'],
            'ER': ['49e3cac631bf487599e3ec0387bf0560', '5ebebc19adf14b5698e8897fe7d8fbde', '77150474837640baa21fd93d8a9ebe80', 'ce32bab6ad9a4a47b03d25e2dbaf9f32', '4d1f067b3c12432ead29617eeaa6d173'],
            'aGEM': None,
            'BIC': ['d422de3abd7d411d8e79e6438203c869', '60907987ce48472791791bd13b96fe1c', '2cd2dfe8d6c84ee8878730fae7a067de', 'a0367c72efe94198a24700acfe2ee89f', '7207fc72b94e48878338d4aaec71dd1c'],
            'ER-ACE': ['8b154fad156b4c0cbbb1b8bc58bcdc3f', '30333191db024696bfa248cc57e92381', '07447ad47d9748ff892ccad06d3c38e7', '02984412768841aeb5b9d51d0c280018', 'eddb1bdfca164937a2ceeef1272ceabd'],
            'GDumb': ['cdd60faa69da45549ed0eb9804c0bbfe', '887164556a234e228bb38017abe44ab6', 'b92f47cfa43d4363a37330ddaf23033c', '81a91133ccda4800ae86b01315b7a0ad', '692af0b23b5d416380421876fc4b591c'],
            'MER': None,
            'MIR': None,
            'DER': ['fd72fc1b81f84b8caeaf7acb207a2f97', 'fa27e138808e48e68f00d23c02b03507', '61dc397fe81f488e98725ee37aacaabe', '7aabdcca65a44d73a5015c15f8cabd25', '217fb45437f4425a94c70017aa26901c'],
            'xDER': ['fe27a2cd3a414b498d984ee9d6019952', '5dc8077a643a40e19b21e6337991b488', 'b5ec01658cf14b81b824ba916782c9a8', '0d7c109accc645e08f1fcd0594f4b6f1', '0f30523a31ad4bedbb831920cb9cd530'],
            'CLeWI': ['0fa6f70ffe104b69ac8899a8ccbacb7f', 'f59a01c79b7a477e8b3866be57ee431b', 'cb276ba0a0f04fe6bfc0be5c5a526110', '4088b64d49814c848ca0f2a2d3e20f51', '75126b5c9c4645beae9204ab7d1ba667'],
        },
        '10 tasks': {
            'Joint': None,
            'Finetuning': None,
            'EWC': None,
            'SI': None,
            'iCARL': ['4dd5033dfda04f6a98b646774b1d8229', 'b3639bf984824afd88e9aec812290dce', '41737907295f43cfa6696761f2455fad', 'e9e2562273844d2f9ac8a8c8ccb507f4', '84de7552699e4beda7196121027622c5'],
            'ER': ['46fda71735f64c8da98d050960dd3cd9', '7835e0307ca143689620fced42f62c9d', '2239aa1376dd4467852af4347adf4811', 'fe4136cbe534423ab257fcf0e8914300', 'be01e5f0683f410885922ef516eb5591'],
            'aGEM': None,
            'BIC': ['1cd4ab0fff654d66952518480bace581', '4396afc6f0ad4da6b95c6e7508f68cc3', '725ad3cc7299453aac8798ffc35676dc', '5d5266c29a80430facfe17080bfb2edb', 'ba359e2d933e471087fea26956cafd6d'],
            'ER-ACE': ['734e25c074514804a0c012e05d9b5450', '1b6cd95859c94471b3cdbed16d242306', '77d24d4c2eeb4b3d8c8566ec5494ac98', '26cd9495cfda4a4984b2532f8d6a80e7', 'eeb81e6993164c8090da796abc8f9ee1'],
            'GDumb': ['8b5ec60afcc14f0489d3861c66082c51', 'e53a5538208946b9b2c21b492eeb2af4', 'b53f529c1d2d4f26927c84445b7309ab', '6205d45f81e040e98680be7af7f07e88', '10c69bba417a49c38b07c8bc6d81fb49'],
            'MER': None,
            'MIR': None,
            'DER': ['ae8b5183dcf2473098e4d6a2c040f8d7', 'a8002e911d8343f99aec7a59d67c34bb', '186e1034cd914a658c83d2c1c076df27', 'ca4ab2fde2a54595b308db566fc2753a', '142ce6e4e98d4c94b050c1f1acc55601'],
            'xDER': ['f248a2f2900b4f0f9682e4629d49930b', 'e7f20f8c086243e29699307e782dd116', '66c73464bf1048cf9d0d454814f6607e', 'b9b03222b2b8470c9cc5303510dda447', '1e2937c86b54484cbf88569da7b00ba9'],
            'CLeWI': ['cf7c6b8b65b04450a468d2f2fa0e9ac9', '5b0b0cec489644b789105d0852223c02', '189b17529d5d49dfb549d1c3d3256306', '66c13b0edaff45eca4c5d79facce9c08', '60ccb27498024749a1292d1ab2ec66e1'],
        },
        '20 tasks': {
            'Joint': None,
            'Finetuning': None,
            'EWC': None,
            'SI': None,
            'iCARL': ['94de55f367924c248a45c5987df33524', 'c8e0f027b51c49648f69f0eb38509462', '610e0073957d49548b68791dd9246919', '7a08a1afe0c741f6a5e893681aa9f8a2', 'de486b95a6c74bfda0a9dd1dd1ec246d'],
            'ER': ['b493ab8e54004750896dbf9cebf0b609', '1951a520b7974b7c9a6d2e26f70f1d3c', '997ecc0f525f4071a74f0b216e9f39c8', 'a8513f0c1db74676af3b9454f49e7a2f', 'e011a0cbc9854328b248df835ae5d178'],
            'aGEM': None,
            'BIC': ['3baf04da53404ca4bbcc0506c90545bb', '5b0b5502cc82427f80df476408ce7da5', '12d168011906473aa728600e672f91d8', '1e686f8e6e4e4c96b113adfe1547e01c', '25c878dddec74378b069173f3caf503f'],
            'ER-ACE': ['a7e438cff0b64590a1b51e421bc200fc', '1b04f1d16d584333826c3c29d277633b', '459fb22963c04042800b2492307c208d', '9f2060e2ea934c109d5447895fddbfcc', 'a41d246aac464288bbcd990e602c276b'],
            'GDumb': ['ff544949cdda41dcb17666474f40db51', '330650c362984bd2abf34d6fbad93366', 'c3e0e9ddc19f4e1c8368411a54b31773', 'd11cb963b54c4a64ba58c50f86651eb6', 'f1389212c51b455eae5d57f8eeac01d0'],
            'MER': None,
            'MIR': None,
            'DER': ['cd06015c7db24f308176ed5a2bb632d9', '6946c90170914e319cd522c2563995a8', '22fb4bca74c0441d8fc7314da250b4ba', 'df57945b9e304e94b6eb76c61e9778ac', '3157516230564a7887a4a4164843e12d'],
            'xDER': ['f4510037fdbf409d9a39c633d9961f33', '6279780737c64700af2dd780637300f4', '29f2ff8dba584728ac379c76e54341ab', '95383a9572b94a0180fa62dfb4bb01b6', '27b6ab60af294ce3a40ac56a33ffbe1b'],
            'CLeWI': ['7157360c5e63454d9dd11a794f7a4319', '98284fb1575e4863a79b1e3a2b9d2ab6', '4cfed79122ab440aa81d5f4f1006b70a', 'bace4c4d02c24be2b923e5d4d8e84289'],
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
