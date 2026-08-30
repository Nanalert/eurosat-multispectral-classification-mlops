"""Shared live MLflow tracking for EuroSAT training scripts."""

import atexit
import copy
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow import MlflowClient
from mlflow.models import infer_signature


class MLflowTracker:
    def __init__(
        self,
        enabled,
        tracking_uri,
        experiment_name,
        run_name,
        parameters,
        tags=None,
    ):
        self.enabled = enabled
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.parameters = parameters
        self.tags = tags or {}
        self.active_run = None
        self.live_logging = True

    def start(self):
        if not self.enabled:
            return
        mlflow.set_tracking_uri(self.tracking_uri)
        try:
            MlflowClient().search_experiments(max_results=1)
        except Exception as error:
            print(
                f"Cannot connect to MLflow at {self.tracking_uri}. "
                f"Training will continue without MLflow: {error}"
            )
            self.enabled = False
            return

        try:
            mlflow.set_experiment(self.experiment_name)
            self.active_run = mlflow.start_run(run_name=self.run_name)
            atexit.register(self.fail)
            mlflow.set_tags({"automatic_tracking": "true", **self.tags})
            mlflow.log_params(self.parameters)
        except Exception as error:
            print(f"Could not start MLflow tracking; training will continue: {error}")
            self.enabled = False
            self.active_run = None

    def log_epoch(
        self,
        epoch,
        train_loss,
        train_accuracy,
        validation_loss,
        validation_accuracy,
    ):
        if not self.enabled:
            return
        if not self.live_logging:
            return
        try:
            mlflow.log_metrics(
                {
                    "train_loss": float(train_loss),
                    "train_accuracy": float(train_accuracy),
                    "validation_loss": float(validation_loss),
                    "validation_accuracy": float(validation_accuracy),
                },
                step=int(epoch),
            )
        except Exception as error:
            self.live_logging = False
            print(
                "MLflow stopped responding. Training will continue and final "
                f"artifacts will be retried after training: {error}"
            )

    def finish(
        self,
        model,
        final_metrics,
        output_directory,
        number_of_classes,
        registered_model_name=None,
        artifact_stem=None,
    ):
        if not self.enabled:
            return
        try:
            mlflow.log_metrics(
                {name: float(value) for name, value in final_metrics.items()}
            )
            output_directory = Path(output_directory)
            if output_directory.exists():
                if artifact_stem is None:
                    mlflow.log_artifacts(output_directory, artifact_path="outputs")
                else:
                    for artifact in output_directory.glob(f"{artifact_stem}*"):
                        if artifact.is_file():
                            mlflow.log_artifact(artifact, artifact_path="outputs")

            export_model = copy.deepcopy(model).cpu().eval()
            input_example = np.zeros((1, 3, 64, 64), dtype=np.float32)
            with torch.inference_mode():
                output_example = export_model(torch.from_numpy(input_example)).numpy()
            signature = infer_signature(input_example, output_example)
            mlflow.pytorch.log_model(
                pytorch_model=export_model,
                name="model",
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
                serialization_format="pt2",
                metadata={
                    "number_of_classes": number_of_classes,
                    "input_format": "normalized NCHW float32 tensor",
                },
            )
            mlflow.end_run(status="FINISHED")
            self.active_run = None
            atexit.unregister(self.fail)
        except Exception:
            self.fail()
            print("MLflow final logging failed; local training outputs were preserved.")

    def fail(self):
        if self.enabled and self.active_run is not None:
            try:
                mlflow.end_run(status="FAILED")
            except Exception as error:
                print(f"Could not mark the MLflow run as failed: {error}")
            finally:
                self.active_run = None
                atexit.unregister(self.fail)
