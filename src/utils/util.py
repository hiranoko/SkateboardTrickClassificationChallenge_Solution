#!/usr/bin/env python
# coding: utf-8


import argparse
import csv
import logging
import os
import random
import time
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

import numpy as np
import requests
import torch
import torch.distributed as dist
from tqdm import tqdm


def make_output_dir_if_needed(output_dir):
    os.makedirs(output_dir / "weight", exist_ok=True)


def line_notify(message):
    line_token = "xxxxxxxx"  # 終わったら無効化する
    endpoint = "https://notify-api.line.me/api/notify"
    message = "\n{}".format(message)
    payload = {"message": message}
    headers = {"Authorization": "Bearer {}".format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)


@contextmanager
def timer(name, logger=None, level=logging.DEBUG):
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    t0 = time.time()
    print_(f"[{name}] start")
    yield
    print_(f"[{name}] done in {time.time() - t0:.0f} s")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Snapshot:
    def __init__(
        self,
        save_best_only=True,
        mode="min",
        initial_metric=None,
        output_dir="",
        name="",
        monitor="metric",
        silent=False,
    ):
        self.save_best_only = save_best_only
        if mode == "min":
            self.mode = 1
        elif mode == "max":
            self.mode = -1
        self.init_metric(initial_metric)
        self.output_dir = Path(output_dir).resolve()
        self.name = name
        self.monitor = monitor
        self.silent = silent
        if not silent:
            print(f"--> Snapshot save in {self.output_dir}")

    def init_metric(self, initial_metric):
        if initial_metric:
            self.best_metric = initial_metric
        else:
            self.best_metric = self.mode * 1000

    def update_best_metric(self, metric):
        if self.mode * metric <= self.mode * self.best_metric:
            self.best_metric = metric
            return 1
        else:
            return 0

    def save_weight_optimizer(self, model, optimizer, prefix):
        torch.save(
            model.state_dict(), self.output_dir / str(self.name + f"_{prefix}.pth")
        )
        # torch.save(optimizer.state_dict(), self.output_dir / str(self.name+f'_optimizer_{prefix}.pt'))

    def snapshot(self, metric, model, optimizer, epoch):
        is_updated = self.update_best_metric(metric)
        if is_updated:
            self.save_weight_optimizer(model, optimizer, "best")
            if not self.silent:
                print(f"--> [best score was updated] {metric}")
        if not self.save_best_only:
            self.save_weight_optimizer(model, optimizer, f"epoch{epoch}")
            print(f"--> [epoch:{epoch}] save shapshot.")


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
