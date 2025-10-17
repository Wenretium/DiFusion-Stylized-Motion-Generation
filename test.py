import json
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf

from mld.callback import ProgressLogger
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger


def print_table(title, metrics):
    table = Table(title=title)

    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(key, str(value))

    console = get_console()
    console.print(table, justify="center")


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER
    logger = create_logger(cfg, phase="test")
    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(OmegaConf.to_yaml(cfg))

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datasets = get_datasets(cfg, phase="testset1")[0]
    cfg.METRIC.FLAG = "TESTSET1"
    cfg.METRIC.TYPE = ['MostMetrics', 'ClassifyMetrics']
    print("datasets module {} initialized".format("".join(
        cfg.TRAIN.DATASETS)))

    # create model
    model = get_model(cfg, datasets)
    print("model {} loaded".format(cfg.model.model_type))
    

    # optimizer
    metric_monitor = {
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose",
    }

    # callbacks
    callbacks = [
        pl.callbacks.RichProgressBar(),
        ProgressLogger(metric_monitor=metric_monitor),
    ]
    print("Callbacks initialized")

    # trainer
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        default_root_dir=cfg.FOLDER_EXP,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=cfg.LOGGER.LOG_EVERY_STEPS,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=None,
        callbacks=callbacks,
    )

    # loading state dict
    print("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))

    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]
    # remove mismatched and unused params
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.split(".")[0] not in ["style_classifier"]:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    
    # strict load classifier model
    if cfg.TESTSET1.NUM_STYLES == 16:
        # assert 'xiabfa' in cfg.TRAIN.PRETRAINED_CLASSIFIER
        PRETRAINED_CLASSIFIER_PATH = 'checkpoints/difusion_16styles_best.ckpt'
        state_dict = torch.load(PRETRAINED_CLASSIFIER_PATH, map_location="cpu")["state_dict"]
        from collections import OrderedDict
        classifier_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.split(".")[0] == "style_classifier":
                name = k.replace("style_classifier.", "")
                classifier_dict[name] = v
        
    elif cfg.TESTSET1.NUM_STYLES == 100:
        # assert 'xiabfa' not in cfg.TRAIN.PRETRAINED_CLASSIFIER
        PRETRAINED_CLASSIFIER_PATH = 'checkpoints/style_encoder_robust47_100.pt' # SmooDi预训练模型
        classifier_dict = torch.load(PRETRAINED_CLASSIFIER_PATH, map_location="cpu")

    model.style_classifier.load_state_dict(classifier_dict, strict=True)


    all_metrics = {}
    replication_times = cfg.TEST.REPLICATION_TIMES
    # calculate metrics
    for i in range(replication_times):
        metrics_type = ", ".join(cfg.METRIC.TYPE)
        print(f"Evaluating {metrics_type} - Replication {i}")
        metrics = trainer.test(model, datamodule=datasets)[0]
        for key, item in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = [item]
            else:
                all_metrics[key] += [item]

    all_metrics_new = {}
    for key, item in all_metrics.items():
        mean, conf_interval = get_metric_statistics(np.array(item),
                                                    replication_times)
        all_metrics_new[key + "/mean"] = mean
        all_metrics_new[key + "/conf_interval"] = conf_interval
    print_table(f"Mean Metrics", all_metrics_new)
    all_metrics_new.update(all_metrics)
    # save metrics to file
    metric_file = output_dir.parent / f"metrics_{cfg.TIME}.json"
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics_new, f, indent=4)
    print(f"Testing done, the metrics are saved to {str(metric_file)}")


if __name__ == "__main__":
    main()
