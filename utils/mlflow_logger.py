import yaml
import os
import pathlib
import tempfile
import torch
import re
import numpy as np
import mlflow
import mlflow.pytorch

import utils.loggers
from utils.metrics import backward_transfer, forward_transfer, forgetting


class MLFlowLogger(utils.loggers.Logger):
    def __init__(self, setting_str: str, dataset_str: str, model_str: str,
                 run_id=None, experiment_name='Default', nested=False, run_name=None):
        super().__init__(setting_str, dataset_str, model_str)
        self.run_id = run_id
        self.experiment_name = experiment_name
        client = mlflow.tracking.MlflowClient()
        self.experiment = client.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            new_id = self.find_last_exp_id(client) + 1
            artifact_location = repo_dir() / 'mlruns' / str(new_id)
            id = mlflow.create_experiment(experiment_name, artifact_location=str(artifact_location))
            self.experiment = client.get_experiment(id)
        self.experiment_id = self.experiment.experiment_id
        self.nested = nested

        if self.run_id == None:
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name, nested=nested):
                active_run = mlflow.active_run()
                self.run_id = active_run.info.run_id

    def find_last_exp_id(self, client):
        last_id = -1
        for i in range(100):
            try:
                client.get_experiment(str(i))
            except:
                break
            last_id = i
        return last_id

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        if self.setting == 'general-continual':
            self.accs.append(mean_acc)
        elif self.setting == 'domain-il':
            mean_acc, _ = mean_acc
            self.accs.append(mean_acc)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            self.accs.append(mean_acc_class_il)
            self.accs_mask_classes.append(mean_acc_task_il)

    def log_fullacc(self, accs):
        if self.setting == 'class-il':
            acc_class_il, acc_task_il = accs
            self.fullaccs.append(acc_class_il)
            self.fullaccs_mask_classes.append(acc_task_il)
            for t, acc in enumerate(acc_class_il):
                self.log_metric(f'acc_class_il_task_{t}', acc)
            for t, acc in enumerate(acc_task_il):
                self.log_metric(f'acc_task_il_task_{t}', acc)

    def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes):
        self.fwt = forward_transfer(results, accs)
        self.log_metric('fwt', self.fwt)
        if self.setting == 'class-il':
            self.fwt_mask_classes = forward_transfer(results_mask_classes, accs_mask_classes)
            self.log_metric('fwt_mask_classes', self.fwt_mask_classes)

    def add_bwt(self, results, results_mask_classes):
        self.bwt = backward_transfer(results)
        self.log_metric('bwt', self.bwt)
        self.bwt_mask_classes = backward_transfer(results_mask_classes)
        self.log_metric('bwt_mask_classes', self.bwt_mask_classes)

    def add_forgetting(self, results, results_mask_classes):
        self.forgetting = forgetting(results)
        self.log_metric('forgetting', self.forgetting)
        self.forgetting_mask_classes = forgetting(results_mask_classes)
        self.log_metric('forgetting_mask_classes', self.forgetting_mask_classes)

    def log_metric(self, metric_name, value):
        with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id, nested=self.nested):
            mlflow.log_metric(metric_name, value)

    def log_artifact(self, artifact_path, name):
        with SwapArtifactUri(self.experiment_id, self.run_id):
            active_run = mlflow.active_run()
            if active_run is not None and active_run.info.run_id == self.run_id:
                mlflow.log_artifact(artifact_path, name)
            else:
                with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id, nested=self.nested):
                    mlflow.log_artifact(artifact_path, name)

    def log_model(self, model: torch.nn.Module):
        with SwapArtifactUri(self.experiment_id, self.run_id):
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = pathlib.Path(tmpdir) / 'model.pth'
                torch.save(model, model_path)
                with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id, nested=self.nested):
                    mlflow.log_artifact(model_path, 'model')

    def log_avrg_accuracy(self):
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(self.run_id)
        run_metrics = run.data.metrics
        test_accs = [acc for name, acc in run_metrics.items() if name.startswith('test_accuracy_task_')]
        test_avrg_acc = sum(test_accs) / len(test_accs)
        client.log_metric(self.run_id, 'avrg_test_acc', test_avrg_acc)


class SwapArtifactUri:
    def __init__(self, experiment_id, run_id):
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.artifact_uri = None

    def __enter__(self):
        repo_path = repo_dir()
        meta_path = repo_path / 'mlruns' / f'{self.experiment_id}' / f'{self.run_id}' / 'meta.yaml'

        run_meta = self.load_meta(meta_path)

        self.artifact_uri = run_meta['artifact_uri']
        run_meta['artifact_uri'] = f'file://{repo_path}/mlruns/{self.experiment_id}/{self.run_id}/artifacts'
        with open(meta_path, 'w') as file:
            yaml.safe_dump(run_meta, file)

    def __exit__(self, exc_type, exc_value, exc_tb):
        repo_path = repo_dir()
        meta_path = repo_path / 'mlruns' / f'{self.experiment_id}' / f'{self.run_id}' / 'meta.yaml'

        run_meta = self.load_meta(meta_path)
        run_meta['artifact_uri'] = self.artifact_uri
        with open(meta_path, 'w') as file:
            yaml.safe_dump(run_meta, file)

    def load_meta(self, meta_path):
        with open(meta_path, 'r') as file:
            run_meta = yaml.safe_load(file)
        return run_meta


def repo_dir():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = pathlib.Path(repo_dir)
    if type(repo_dir) == pathlib.WindowsPath:
        repo_dir = pathlib.Path(*repo_dir.parts[1:]).as_posix()
    return repo_dir.parent
