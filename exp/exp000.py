import os
import math
import joblib
import argparse
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Literal
import pprint

import cv2

import numpy as np
import pandas as pd

from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

import timm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


config: DictConfig = OmegaConf.create(
    dict(
        train=True,
        train_fold=[0, 1, 2, 3, 4],
        inference=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=2022,
        epocs=50,
        img_size=768,
        n_splits=5,
        num_classes=15587,
        data_path="./input/happy-whale-and-dolphin",
        output_path="./output",
        log_path="wandb_logs",
        wandb_project="",
        model=dict(
            name="tf_efficientnet_b4_ns",
            pretrained=True,
            arc_params=dict(
                s=30.0,
                m=0.30,
                ls_eps=0.0,
                easy_margin=False
            )
        ),
        trainer=dict(
            gpus=1,
            accumulate_grad_batches=1,
            progress_bar_refresh_rate=1,
            fast_dev_run=False,
            num_sanity_val_steps=3,
            resume_from_checkpoint=None,
            precision=16,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            limit_test_batches=0.0,
            check_val_every_n_epoch=1,
            log_evary_n_steps=10,
            flush_logs_every_n_steps=10,
            profiler="simple",
            deterministic=False,
            weights_summary="top",
            reload_dataloaders_every_epoch=True,
        ),
        train_loader=dict(
            batch_size=4,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ),
        val_loader=dict(
            batch_size=4,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        ),
        test_loader=dict(
            batch_size=4,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        ),
        optimizer=dict(
            name="optim.AdamW",
            params=dict(
                lr=1e-4
            )
        ),
        scheduler=dict(
            name="optim.lr_scheduler.CosineAnnealingLR",
            params=dict(
                T_max=500,
            )
        ),
        loss="nn.CrossEntropyLoss",
        callbacks=dict(
            monitor_metric="val_loss",
            mode="min",
            patience=10,
        )
    )
)

torch.autograd.set_detect_anomaly(True)
seed_everything(config.seed)


def get_df(config: DictConfig) -> pd.DataFrame:
    train_img_dir = Path(config.data_path) / "train_images"

    def _get_train_file_path(idx: str) -> Path:
        return train_img_dir / idx

    df = pd.read_csv(Path(config.data_path) / "train.csv")
    df.loc[:, "file_path"] = df["image"].map(_get_train_file_path)
    return df


def encode_ids(df: pd.DataFrame, config: DictConfig, le_encoder: LabelEncoder, save: bool = False) -> pd.DataFrame:
    df.loc[:, "individual_id"] = le_encoder.fit_transform(df["individual_id"])
    if save:
        pickle_data_path = Path(config.output_path) / "le.pkl"
        with pickle_data_path.open("wb") as f:
            joblib.dump(le_encoder, f)
    return df


def create_folds(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df["individual_id"])):
        df.loc[val_idx, "fold"] = fold
    df["fold"] = df["fold"].astype(int)
    return df


def preprocess_df(
    config: DictConfig, le_encoder: LabelEncoder, recache: bool = False, save: bool = True
) -> pd.DataFrame:
    train_df_path = Path(config.data_path) / f"train-df-fold{config.n_splits}-seed{config.seed}.csv"
    if not recache and train_df_path.exists():
        df = pd.read_csv(train_df_path)
        return df

    df = get_df(config)
    df = encode_ids(df, config, le_encoder, save)
    df = create_folds(df, config)
    return df


def get_transform(config: DictConfig) -> dict:
    transform = {
        "train": A.Compose(
            [
                A.Resize(config.img_size, config.img_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0
                ),
                ToTensorV2()
            ],
            p=1.0
        ),
        "val": A.Compose(
            [
                A.Resize(config.img_size, config.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0
                ),
                ToTensorV2()
            ],
            p=1.0
        ),
    }
    return transform


# =============================
# Data
# =============================
class HappyWhaleDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, mode: str, transform: Optional[dict] = None
    ) -> None:
        assert mode in {"train", "val", "test"}
        super().__init__()
        self._df = df
        self._file_names = df["file_path"].values
        self._labels = df["individual_id"].values

        self._mode = mode
        self._transform = transform

        self._dtype = torch.float if config.trainer.precision == 32 else torch.half

    def __len__(self) -> int:
        return len(self._df)

    def __get_img(self, img_path: str) -> np.ndarray:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx: int) -> dict:
        img_path = self._file_names[idx]
        img = self.__get_img(img_path)
        label = self._labels[idx]

        if self._transform:
            img = self._transform[self._mode](image=img)["image"]

        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)

        if self._mode == "test":
            return {
                    "image": img.to(dtype=self._dtype)
            }

        return {
            "image": img.to(dtype=self._dtype),
            "label": torch.tensor(label, dtype=self._dtype)
        }


class MyLitDataModule(pl.LightningDataModule):
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, config: DictConfig) -> None:
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._test_df = test_df
        self._config = config

    def __create_dataset(self, mode: str) -> Dataset:
        if mode == "train":
            return HappyWhaleDataset(df=self._train_df, transform=get_transform(self._config), mode="train")
        elif mode == "val":
            return HappyWhaleDataset(df=self._val_df,  transform=get_transform(self._config), mode="val")
        elif mode == "test":
            return HappyWhaleDataset(df=self._test_df,  transform=get_transform(self._config), mode="test")
        else:
            raise ValueError

    def train_dataloader(self) -> DataLoader:
        dataset = self.__create_dataset(mode="train")
        return DataLoader(dataset, **self._config.train_loader)

    def val_dataloader(self) -> DataLoader:
        dataset = self.__create_dataset(mode="val")
        return DataLoader(dataset, **self._config.val_loader)

    def test_dataloader(self) -> DataLoader:
        dataset = self.__create_dataset(mode="test")
        return DataLoader(dataset, **self._config.test_loader)

# =============================
# Model
# =============================
class GeM(nn.Module):
    """ GeM Pooling

    Ref:
        * https://www.kaggle.com/vladvdv/pytorch-train-notebook-arcface-gem-pooling#GeM-Pooling
        * https://amaarora.github.io/2020/08/30/gempool.html

    """
    def __init__(self, p: int = 3, eps: float = 1e-6) -> None:
        super().__init__()
        self._p = nn.Parameter(torch.ones(1) * p)
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gem(x, p=self._p, eps=self._eps)

    def gem(self, x: torch.Tensor, p: nn.parameter.Parameter, eps: float = 1e-6) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
                "(" + f"p={self._p.data.tolist()[0]:.4f}" + \
                f",eps={self.eps})"


class ArcMarginProduct(nn.Module):
    """ Implement of large margin arc distance

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input features
        m: margin cos(theta + m)

    Ref:
        https://www.kaggle.com/vladvdv/pytorch-train-notebook-arcface-gem-pooling#ArcFace
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: DictConfig,
        s: float = 30.0,
        m: float = 0.50,
        easy_margin: bool = False,
        ls_eps: float = 0.0
    ) -> None:
        super().__init__()
        self.in_featurs = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self._device = torch.device(config.device)
        self._dtype = torch.float if config.trainer.precision == 32 else torch.half

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # cos(theta) & phi(theta)
        cos = F.linear(F.normalize(x), F.normalize(self.weight))
        sin = torch.sqrt(1.0 - torch.pow(cos, 2))
        phi = cos * self.cos_m - sin * self.sin_m
        phi = phi.to(dtype=self._dtype)

        if self.easy_margin:
            phi = torch.where(cos > 0.0, phi, cos)
        else:
            phi = torch.where(cos > self.th, phi, cos - self.mm)

        # conver label to one-hot
        one_hot = torch.zeros(cos.size(), device=self._device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cos)
        output *= self.s
        return output


class HappyWhaleModel(nn.Module):
    def __init__(self, model_name: str, config: DictConfig, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pooling = GeM()
        self.drop = nn.Dropout(p=0.2, inplace=False)
        self.fc = nn.Linear(in_features, 512)
        self.arc = ArcMarginProduct(
            512,
            config.num_classes,
            config=config,
            s=config.model.arc_params.s,
            m=config.model.arc_params.m,
            easy_margin=config.model.arc_params.easy_margin,
            ls_eps=config.model.arc_params.ls_eps
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        pooled_features = self.pooling(features).flatten(1)
        pooled_drop = self.drop(pooled_features)
        emb = self.fc(pooled_drop)
        output = self.arc(emb, labels)
        return output, emb


class MyLitModel(pl.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self._config = config
        self._criterion = eval(config.loss)()
        self._transform = get_transform(config)
        self._dtype = torch.float if config.trainer.precision == 32 else torch.half
        self.__build_model()
        self.save_hyperparameters(config)

    def __build_model(self) -> None:
        self.model = HappyWhaleModel(
            model_name=self._config.model.name,
            config=self._config,
            pretrained=self._config.model.pretrained,
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(images, labels)

    def training_step(self, batch, batch_idx: int):
        loss, pred, labels = self.__share_step(batch, "train")
        return {"loss": loss, "pred": pred, "labels": labels}

    def validation_step(self, batch, batch_idx: int):
        loss, pred, labels = self.__share_step(batch, "val")
        return {"pred": pred, "labels": labels}

    def __share_step(self, batch, mode):
        images, labels = batch["image"].to(dtype=self._dtype), batch["label"].to(dtype=torch.long)
        outputs, emb = self.forward(images, labels)
        outputs = outputs.to(dtype=self._dtype)
        # print(f"output: {outputs.dtype}, labels: {labels.dtype}")
        # print(f"output: {outputs}, labels: {labels}")
        loss = self._criterion(outputs, labels)

        pred = outputs.sigmoid(dim=1).detach().cpu()
        labels = labels.detach().cpu()
        return loss, pred, labels

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "val")

    def __share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out["pred"], out["labels"]
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        metrics = torch.sqrt(((labels - preds) ** 2).mean())
        self.log(f"{mode}_loss", metrics)

    def configure_optimizers(self):
        optimizer = eval(self._config.optimizer.name)(
            self.parameters(), **self._config.optimizer.params
        )
        scheduler = eval(self._config.scheduler.name)(
            optimizer, **self._config.scheduler.params
        )
        return [optimizer], [scheduler]

    import argparse


def train(model, datamodule, fold: int, config: DictConfig) -> None:
    # instanciate callbacks
    earystopping = EarlyStopping(
        monitor=config.callbacks.monitor_metric,
        patience=config.callbacks.patience,
        verbose=True,
        mode=config.callbacks.mode
    )
    lr_monitor = callbacks.LearningRateMonitor()
    loss_checkpoint = callbacks.ModelCheckpoint(
        dirpath=f"./output/{config.model.name}",
        filename=f"{config.model.name}" + "-{epoch}-{step}",
        monitor=config.callbacks.monitor_metric,
        save_top_k=1,
        mode=config.callbacks.mode,
        save_weights_only=False
    )

    # instanciate logger
    pl_logger = WandbLogger(
        name=config.model.name + f"_fold{config.n_splits}",
        save_dir=config.log_path,
        project=config.wandb_project,
        version=hashlib.sha224(bytes(str(dict(config)), "utf8")).hexdigest()[:4],
        anonymous=True,
        group=config.model.name,
        # tags=[config.labels]
    )

    # warm start
    if config.warm_start_path is not None:
        checkpoint_path = Path(config.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"{config.checkpoint_path} does not exist")
        logger.info(checkpoint_path)
        config.trainer.resume_from_checkpoint = checkpoint_path

    trainer = pl.Trainer(
        logger=pl_logger,
        max_epochs=config.epoch,
        callbacks=[lr_monitor, loss_checkpoint, earystopping],
        **config.trainer,
    )
    trainer.fit(model, datamodule=datamodule)


def update_config(config: DictConfig) -> "argparse.NameSpace":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fold", default=-1, type=int, nargs="*")
    parser.add_argument("--tpu")
    parser.add_argument("--tpu_cores", default=-10, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--exp", action="store_true")
    args = parser.parse_args()

    if args.train_fold != -1:
        config["train_fold"] = args.train_fold
        print("train_epoch is specified: ", config["train_fold"])

    if args.tpu:
        config["trainer"]["tpu_cores"] = args.tpu_cores
        config["trainer"].pop("gpus")
        print("tpu is specified with the number of", config["trainer"]["tpu_cores"])

    if args.exp:
        print(" ####### exp mode is called. ####### ")
        config["trainer"]["limit_train_batches"] = 0.5
        config["trainer"]["limit_val_batches"] = 0.5

    if args.debug:
        print(" ####### debug mode is called. ####### ")
        config["trainer"]["limit_train_batches"] = 0.005
        config["trainer"]["limit_val_batches"] = 0.005
        config["epoch"] = 1

    return config


def main(config: DictConfig) -> None:


    config = update_config(config)
    pprint.pprint(config)

    for fold in range(config.n_splits):
        # prepare data
        le_encoder = LabelEncoder()
        df = preprocess_df(config, le_encoder)
        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]
        datamodule = MyLitDataModule(train_df, val_df, config)
        model = MyLitModel(config)
        if config.train and fold in config.train_fold:
            print("#" * 8 + f"  Fold: {fold}  " + "#" * 8)
            train(model, datamodule, fold, config)

    if config.inference:
        # inference関数を実装する
        checkpoint_path = config.trainer.resume_from_checkpoint
        state_dict = torch.load(checkpoint_path)["state_dict"]
        


if __name__ == "__main__":
    main(config)
