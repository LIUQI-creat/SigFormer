from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from openpack_toolkit import OPENPACK_OPERATIONS
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics.functional import f1_score

from utils.loss import FocalLoss, OpWeightedKappaLoss,BoundaryRegressionLoss
from utils.model import SigFormer
logger = getLogger(__name__)


class TransformerPL(optorch.lightning.BaseLightningModule):
    def init_model(self, cfg: DictConfig) -> torch.nn.Module:
        dstream_conf = self.cfg.dataset.stream
        in_ch = len(dstream_conf.devices) * 3

        # Edit here to use your custom model!
        if cfg.model.name == "SigFormer":
            model = SigFormer(
                num_classes=len(OPENPACK_OPERATIONS),
                cfg=cfg,
            )
        
        self.cfg = cfg
        return model

    def set_fold(self, fold):
        self.fold = fold
        self.log_dict: Dict[str, List] = {"train": [], "val": [], "test": []}
        # log_dictはクラスメンバ（？）だからインスタンスを変えても同じものになってしまう、上記で上書き

    def init_criterion(self, cfg: DictConfig):
        ignore_cls = [(i, c) for i, c in enumerate(cfg.dataset.classes.classes) if c.is_ignore]

        if cfg.train.ignore_index:
            ignore_index = ignore_cls[-1][0]
        else:
            ignore_index = -100

        if cfg.loss.focal_loss_gamma > 0:
            criterion = FocalLoss(
                ignore_index=ignore_index,
                label_smoothing=cfg.train.label_smoothing,
                gamma=cfg.loss.focal_loss_gamma,
            )
        else:
            criterion = torch.nn.CrossEntropyLoss(
                ignore_index=ignore_index, label_smoothing=cfg.train.label_smoothing
            )
        self.criterion_test = torch.nn.CrossEntropyLoss(ignore_index=ignore_cls[-1][0], label_smoothing=0.0)
        self.criterion_w_kappa = OpWeightedKappaLoss(
            num_classes=11, ignore_index=ignore_index, weightage=cfg.loss.w_kappa_loss_weightage
        )
        self.boundary_loss = BoundaryRegressionLoss()
        return criterion

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        if self.cfg.train.dataaug.mixup_p > 0:
            batch, label_a, label_b, lam = mixup(
                batch, alpha=self.cfg.train.dataaug.mixup_alpha, p=self.cfg.train.dataaug.mixup_p
            )
        if self.cfg.train.dataaug.shuffle_p > 0:
            batch = shuffle_patch(
                batch, alpha=self.cfg.train.dataaug.shuffle_alpha, p=self.cfg.train.dataaug.shuffle_p
            )

        (
            x_imu,
            x_keypoint,
            x_e4acc,
            x_bbox,
            x_ht,
            x_printer,
            t,
        ) = self._split_data(batch)

        if self.cfg.twostep_pretrain.use and self.current_epoch < self.cfg.twostep_pretrain.pretrain_epoch:
            y_imu, y_keypoint, y_kinect = self.net.forward_pretrain(
                x_imu,
                x_keypoint,
                x_e4acc,
                x_bbox,
                x_ht,
                x_printer,
            )
            y_hat = torch.stack([y_imu, y_keypoint, y_kinect]).mean(0)
            loss = (
                self.criterion(y_imu.flatten(0, 1), t.flatten(0, 1))
                + self.criterion(y_keypoint.flatten(0, 1), t.flatten(0, 1))
                + self.criterion(y_kinect.flatten(0, 1), t.flatten(0, 1))
            )
        else:
            y_hat, inner_logits,bound,inner_bound = self.net(
                x_imu,
                x_keypoint,
                x_e4acc,
                x_bbox,
                x_ht,
                x_printer,
            )
            if self.cfg.train.dataaug.mixup_p > 0:
                label_b.to(device=self.device, dtype=torch.long)
                loss = lam * self.criterion(y_hat.flatten(0, 1), t.flatten(0, 1)) + (
                    1 - lam
                ) * self.criterion(y_hat.flatten(0, 1), label_b.flatten(0, 1))
            else:
                loss = self.criterion(y_hat.flatten(0, 1), t.flatten(0, 1))
            if self.cfg.train.inner_loss > 0:
                loss = (1 - self.cfg.train.inner_loss) * loss + self.cfg.train.inner_loss * self.criterion(
                    inner_logits.flatten(0, 1), t.flatten(0, 1)
                )
        if self.cfg.loss.w_kappa_loss != 0:
            loss = loss + self.cfg.loss.w_kappa_loss * self.criterion_w_kappa(
                y_hat.flatten(0, 1), t.flatten(0, 1)
            )
            
        # BoundaryRegressionLoss
#         boundaryloss = 0.0
#         inner_boundaryloss = 0.0
        if self.cfg.loss.boundary != 0:
#             n = len(bound)
#             for out in bound:
#                 boundaryloss += self.cfg.loss.boundary * self.boundary_loss(out.flatten(0,1),t.flatten(0, 1)) / n
#             #print('boundary',boundaryloss.item())
#             loss += boundaryloss
            boundaryloss = self.cfg.loss.boundary * self.boundary_loss(bound.flatten(0,1),t.flatten(0, 1))
            loss += boundaryloss
            
        if self.cfg.loss.inner_boundary != 0:
            inner_boundaryloss = self.cfg.loss.inner_boundary * self.boundary_loss(inner_bound.flatten(0, 1),t.flatten(0, 1))
            #print('inner_boundary',inner_boundaryloss.item())
            loss += inner_boundaryloss
                
#             boundaryloss = self.cfg.loss.boundary * self.boundary_loss(bound.flatten(0, 1),t.flatten(0, 1))
#             print(boundaryloss.item())
#             loss += boundaryloss
        bloss = boundaryloss + inner_boundaryloss
        acc = self.calc_accuracy(y_hat.transpose(1, 2), t)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("train_boundaryloss", bloss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return {"loss": loss, "acc": acc, "boundaryloss": bloss}

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0) -> Dict:
        with torch.inference_mode():
            (
                x_imu,
                x_keypoint,
                x_e4acc,
                x_bbox,
                x_ht,
                x_printer,
                t,
            ) = self._split_data(batch)
            if (
                self.cfg.twostep_pretrain.use
                and self.current_epoch < self.cfg.twostep_pretrain.pretrain_epoch
            ):
                y_imu, y_keypoint, y_kinect = self.net.forward_pretrain(
                    x_imu,
                    x_keypoint,
                    x_e4acc,
                    x_bbox,
                    x_ht,
                    x_printer,
                )
                y_hat = torch.stack([y_imu, y_keypoint, y_kinect]).mean(0)
                loss = (
                    self.criterion_test(y_imu.flatten(0, 1), t.flatten(0, 1))
                    + self.criterion_test(y_keypoint.flatten(0, 1), t.flatten(0, 1))
                    + self.criterion_test(y_kinect.flatten(0, 1), t.flatten(0, 1))
                )
            else:
                y_hat, feat,bound, inner_bound = self.net(
                    x_imu,
                    x_keypoint,
                    x_e4acc,
                    x_bbox,
                    x_ht,
                    x_printer,
                )
                loss = self.criterion_test(y_hat.flatten(0, 1), t.flatten(0, 1))
                testbloss = self.boundary_loss(bound.flatten(0,1),t.flatten(0, 1))
                loss += testbloss
            # valではCEのみ
            # loss = loss + self.cfg.train.w_kappa_loss * self.criterion_w_kappa(
            #     y_hat.flatten(0, 1), t.flatten(0, 1)
            # )
            acc = self.calc_accuracy(y_hat.transpose(1, 2), t)

            if batch_idx == 0:
                self.val_y_batch = []
                self.val_t_batch = []
            self.val_y_batch.append(y_hat.transpose(1, 2))
            self.val_t_batch.append(t)
        return {"loss": loss, "acc": acc, "boundaryloss": testbloss}

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        (
            x_imu,
            x_keypoint,
            x_e4acc,
            x_bbox,
            x_ht,
            x_printer,
            t,
        ) = self._split_data(batch)
        ts_unix = batch["ts"]

        y_hat, feat,bound,inner_bound = self.net(
            x_imu,
            x_keypoint,
            x_e4acc,
            x_bbox,
            x_ht,
            x_printer,
        )

        outputs = dict(t=t, y=y_hat.transpose(1, 2), boundary = bound.transpose(1,2),unixtime=ts_unix)
        return outputs

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.cfg.train.optimizer.decrease_lr:
            parameters = []
            for name, module in self.net.named_children():
                if name in ["embedding", "linear_head"]:
                    parameters.append(
                        {
                            "params": module.parameters(),
                            "lr": self.cfg.train.optimizer.lr / self.cfg.model.num_patches,
                        }
                    )
                else:
                    parameters.append(
                        {
                            "params": module.parameters(),
                        }
                    )
        else:
            parameters = self.parameters()

        if self.cfg.train.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(
                parameters,
                lr=self.cfg.train.optimizer.lr,
                weight_decay=self.cfg.train.optimizer.weight_decay,
            )
        elif self.cfg.train.optimizer.type == "SGD":
            optimizer = torch.optim.SGD(
                parameters,
                lr=self.cfg.train.optimizer.lr,
                weight_decay=self.cfg.train.optimizer.weight_decay,
                momentum=0.9,
            )
        elif self.cfg.train.optimizer.type == "RAdam":
            optimizer = torch.optim.RAdam(
                parameters,
                lr=self.cfg.train.optimizer.lr,
                weight_decay=self.cfg.train.optimizer.weight_decay,
            )
        elif self.cfg.train.optimizer.type == "AdamW":
            optimizer = torch.optim.AdamW(
                parameters,
                lr=self.cfg.train.optimizer.lr,
                weight_decay=self.cfg.train.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"{self.cfg.train.optimizer.type} is not supported.")

        if self.cfg.train.optimizer.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.train.optimizer.cosine_step,
                eta_min=self.cfg.train.optimizer.lr * 0.1,
            )
        elif self.cfg.train.optimizer.lr_scheduler == "multi_step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.cfg.train.optimizer.multistep_milestones,
                gamma=self.cfg.train.optimizer.multistep_gamma,
            )
        elif self.cfg.train.optimizer.lr_scheduler == "ori_warmup":
            scheduler = OriWarmupScheduler(
                optimizer=optimizer, warmup_epochs=self.cfg.train.optimizer.warmup_step
            )
        else:
            raise ValueError(f"{self.cfg.train.optimizer.lr_scheduler} is not supported.")
        return [optimizer,], [
            scheduler,
        ]

    def training_epoch_end(self, outputs):
        log = dict()
        #print('log_keys',self.log_keys)
        for key in self.log_keys:
            vals = [x[key] for x in outputs if key in x.keys()]
            if len(vals) > 0:
                avg = torch.stack(vals).mean().item()
                log[f"train/{key}"] = avg
                self.log(
                    f"{key}/train/fold-{self.fold}",
                    avg,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
        self.log_dict["train"].append(log)

    def validation_epoch_end(self, outputs):
        if isinstance(outputs[0], list):
            # When multiple dataloader is used.
            _outputs = []
            for out in outputs:
                _outputs += out
            outputs = _outputs

        log = dict()
        for key in self.log_keys:
            vals = [x[key] for x in outputs if key in x.keys()]
            if len(vals) > 0:
                avg = torch.stack(vals).mean().item()
                log[f"val/{key}"] = avg
                self.log(
                    f"{key}/val/fold-{self.fold}",
                    avg,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

        val_y = torch.concat(self.val_y_batch)
        val_t = torch.concat(self.val_t_batch)
        f1macro = self.calc_f1macro(val_y, val_t)
        log["val/f1macro"] = f1macro
        self.log(
            f"f1macro/val/fold-{self.fold}",
            f1macro,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.log_dict["val"].append(log)

        self.print_latest_metrics()

        if len(self.log_dict["val"]) > 0:
            val_loss = self.log_dict["val"][-1].get("val/loss", None)
            self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            val_acc = self.log_dict["val"][-1].get("val/acc", None)
            self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            val_bloss = self.log_dict["val"][-1].get("val/boundaryloss", None)
            self.log("val_bloss", val_bloss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_f1macro", f1macro, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def _split_data(self, batch):
        x_imu = batch["imu"].to(device=self.device, dtype=torch.float)
        x_keypoint = batch["keypoint"].to(device=self.device, dtype=torch.float)
        x_e4acc = batch["e4acc"].to(device=self.device, dtype=torch.float)
        x_bbox = batch["bbox"].to(device=self.device, dtype=torch.float)
        x_ht = batch["ht"].to(device=self.device, dtype=torch.int)
        x_printer = batch["printer"].to(device=self.device, dtype=torch.int)
        t = batch["label"].to(device=self.device, dtype=torch.long)

        return (
            x_imu,
            x_keypoint,
            x_e4acc,
            x_bbox,
            x_ht,
            x_printer,
            t,
        )

    def calc_f1macro(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        preds = F.softmax(y, dim=1)
        (batch_size, num_classes, window_size) = preds.size()
        preds_flat = preds.permute(1, 0, 2).reshape(num_classes, batch_size * window_size)
        t_flat = t.reshape(-1)

        ignore_index = num_classes - 1
        f1macro = f1_score(
            preds_flat.transpose(0, 1),
            t_flat,
            average="macro",
            num_classes=num_classes,
            ignore_index=ignore_index,
            task = "multiclass"
        )
        return f1macro

    def print_latest_metrics(self) -> None:
        # -- Logging --
        train_log = self.log_dict["train"][-1] if len(self.log_dict["train"]) > 0 else dict()
        val_log = self.log_dict["val"][-1] if len(self.log_dict["val"]) > 0 else dict()
        log_template = (
            "Epoch[{epoch:0=3}]"
            " TRAIN: loss={train_loss:>7.4f}, boundaryloss={train_bloss:>7.4f}, acc={train_acc:>7.4f}"
            " | VAL: loss={val_loss:>7.4f}, boundaryloss={val_bloss:>7.4f}, acc={val_acc:>7.4f}, f1macro={val_f1macro:>7.4f}"
        )
        logger.info(
            log_template.format(
                epoch=self.current_epoch,
                train_loss=train_log.get("train/loss", -1),
                train_bloss=train_log.get("train/boundaryloss", -1),
                train_acc=train_log.get("train/acc", -1),
                val_loss=val_log.get("val/loss", -1),
                val_bloss=val_log.get("val/boundaryloss", -1),
                val_acc=val_log.get("val/acc", -1),
                val_f1macro=val_log.get("val/f1macro", -1),
            )
        )


def mixup(
    batch,
    alpha=0.8,
    x_keys=[
        "imu",
        "keypoint",
        "ht",
        "printer",
        "kinect_depth",
        "exist_data_kinect_depth",
        "rs02_depth",
        "exist_data_rs02_depth",
    ],
    p=0.2,
):
    lam = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha])).sample().item()
    batch_size = len(batch["label"])
    index = torch.randperm(batch_size)[: int(batch_size * p)]
    index = torch.concat([index, torch.arange(int(batch_size * p), batch_size, 1)])

    for key in x_keys:
        if "exist" in key:
            batch[key] = batch[key].to(torch.int)

    label_a = batch["label"]
    label_b = batch["label"][index]

    mix_batch = {key: lam * batch[key] + (1 - lam) * batch[key][index] for key in x_keys}
    batch.update(mix_batch)

    return batch, label_a, label_b, lam


def _shuffle_patch(batch, alpha=0.1, p=0.2):
    batch_size, patch_length = batch["label"].shape
    shuffle_length = int(patch_length * alpha)
    shuffle_patch_source_ind = torch.randint(0, patch_length, (int(batch_size * p),))
    shuffle_patch_target_ind = torch.randint(0, patch_length - shuffle_length, (int(batch_size * p),))
    shuffle_batch_ind = torch.randint(0, batch_size, (int(batch_size * p),))

    for key in batch.keys():
        for b_ind, source_ind, target_ind in zip(
            shuffle_batch_ind, shuffle_patch_source_ind, shuffle_patch_target_ind
        ):
            source_ind_list = torch.logical_and(
                torch.arange(patch_length) >= source_ind,
                torch.arange(patch_length) < source_ind + shuffle_length,
            )
            tmp = batch[key][b_ind][source_ind_list]
            del_source = batch[key][b_ind][torch.logical_not(source_ind_list)]
            batch[key][b_ind] = torch.concat([del_source[:target_ind], tmp, del_source[target_ind:]])

    return batch


def shuffle_patch(batch, alpha=0.1, p=0.2):
    batch_size, patch_length = batch["label"].shape
    change_label_ind = np.full((batch_size, patch_length), False)
    for b in range(batch_size):
        t = batch["label"][b][0] - 1
        for i, a in enumerate(batch["label"][b]):
            if a != t:
                t = a
                change_label_ind[b, i] = True

    shuffle_source_start_ind = np.zeros(batch_size, dtype=int)
    shuffle_source_end_ind = np.zeros(batch_size, dtype=int)
    for b in range(batch_size):
        _change_label_ind = np.where(change_label_ind[b] == True)[0]
        shuffle_source_start_ind[b] = np.random.choice(_change_label_ind, 1)
        _end_ind = np.where(_change_label_ind > shuffle_source_start_ind[b])[0]
        if len(_end_ind) == 0:
            shuffle_source_end_ind[b] = 49
        else:
            shuffle_source_end_ind[b] = _change_label_ind[_end_ind[0]] - 1

    for b in range(batch_size):
        if torch.rand(1) < p:
            continue
        start = shuffle_source_start_ind[b]
        end = shuffle_source_end_ind[b]
        source_ind_list = torch.logical_and(
            torch.arange(patch_length) >= start,
            torch.arange(patch_length) <= end,
        )
        max_target_ind = patch_length - (end - start) - 2
        if max_target_ind <= 0:
            continue
        target_ind = torch.randint(torch.tensor(0), torch.tensor(max_target_ind), (1,))
        for key in batch.keys():
            tmp = batch[key][b][source_ind_list]
            del_source = batch[key][b][torch.logical_not(source_ind_list)]
            batch[key][b] = torch.concat([del_source[:target_ind], tmp, del_source[target_ind:]])

    return batch


class OriWarmupScheduler(_LRScheduler):
    """TransformerLR class for adjustment of learning rate.

    The scheduling is based on the method proposed in 'Attention is All You Need'.
    """

    def __init__(self, optimizer, warmup_epochs, last_epoch=-1, verbose=False):
        """Initialize class."""
        self.warmup_epochs = warmup_epochs
        self.normalize = self.warmup_epochs**0.5
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Return adjusted learning rate."""
        step = self.last_epoch + 1
        scale = self.normalize * min(step**-0.5, step * self.warmup_epochs**-1.5)
        return [base_lr * scale for base_lr in self.base_lrs]
