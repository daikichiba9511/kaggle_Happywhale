""" exp001

use backfin data as train data


"""
import argparse
import hashlib
import math
import os
import pprint
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdb
from tqdm.auto import tqdm
import faiss
import albumentations as A
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from loguru import logger
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, normalize
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Literal

config: DictConfig = OmegaConf.create(
    dict(
        train=True,
        debug=False,
        train_fold=[0, 1, 2, 3, 4],
        inference=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=2022,
        img_size=224,
        n_splits=5,
        num_classes=15587,
        data_path="./input/happy-whale-and-dolphin",
        output_path="./output",
        log_path="wandb_logs",
        warm_start_path=None,
        wandb_project="HappyWhale",
        model=dict(
            name="tf_efficientnet_b4_ns",
            pretrained=True,
            embedding_size=512,
            arc_params=dict(s=20.0, m=0.50, ls_eps=0.0, easy_margin=False),
        ),
        trainer=dict(
            gpus=1,
            accumulate_grad_batches=4,
            progress_bar_refresh_rate=1,
            fast_dev_run=False,
            num_sanity_val_steps=3,
            resume_from_checkpoint=None,
            precision=16,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            limit_test_batches=0.0,
            check_val_every_n_epoch=1,
            flush_logs_every_n_steps=10,
            profiler="simple",
            deterministic=False,
            benchmark=False,
            weights_summary="top",
            reload_dataloaders_every_epoch=True,
            auto_scale_batch_size=True,
            auto_lr_find=True,
            max_epochs=50,
            stochastic_weight_avg=False,
        ),
        batch_size=128,
        train_loader=dict(shuffle=True, num_workers=2, pin_memory=True, drop_last=True),
        val_loader=dict(
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        ),
        test_loader=dict(
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        ),
        # optimizer=dict(name="optim.AdamW", params=dict(lr=5e-4)),
        optimizer=dict(
            name="optim.AdamW", default_lr=3e-4, params=dict(weight_decay=5e-5)
        ),
        scheduler=dict(
            name="optim.lr_scheduler.CosineAnnealingLR",
            params=dict(T_max=50),
        ),
        loss="nn.CrossEntropyLoss",
        callbacks=dict(
            monitor_metric="val_loss",
            mode="min",
            patience=10,
        ),
    )
)

torch.autograd.set_detect_anomaly(True)
seed_everything(config.seed)

# ============
# Functions
# ============

# ##########
# Metrics
# Ref
# https://www.kaggle.com/pestipeti/explanation-of-map5-scoring-metric
# ##########
def map_per_image(label: str, predictions: List[str]) -> float:
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0


def map_per_set(labels: List[str], predictions: List[List[str]]) -> np.floating:
    """Computes the average over multiple images.

    Parameters
    ----------
    labels : list
             A list of the true labels. (Only one true label per images allowed!)
    predictions : list of list
             A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    return np.mean(
        [map_per_image(label, pred) for label, pred in zip(labels, predictions)]
    )


def plot_dist(
    label: np.ndarray, save_name: str, figsize: Tuple[int, int] = (400, 200)
) -> None:
    plt.figure(figsize=figsize)
    plt.hist(label, label="label")
    plt.savefig(save_name)


def get_df(config: DictConfig, mode: str = "train") -> pd.DataFrame:
    assert mode in {"train", "test"}
    if mode == "train":
        train_img_dir = Path(config.data_path + "-backfin") / "train_images"

        def _get_train_file_path(idx: str) -> Path:
            return train_img_dir / idx

        # use backfin data
        train_df = pd.read_csv(Path(config.data_path + "-backfin") / "train.csv")
        train_df.loc[:, "file_path"] = train_df["image"].map(_get_train_file_path)
        train_df["species"].replace(
            {
                "globis": "short_finned_pilot_whale",
                "pilot_whale": "short_finned_pilot_whale",
                "kiler_whale": "killer_whale",
                "bottlenose_dolpin": "bottlenose_dolphin",
            },
            inplace=True,
        )
        return train_df

    elif mode == "test":
        test_img_dir = Path(config.data_path) / "test_images"

        def _get_test_file_path(idx: str) -> Path:
            return test_img_dir / idx

        test_df = pd.read_csv(Path(config.data_path) / "sample_submission.csv")
        test_df.loc[:, "file_path"] = test_df["image"].map(_get_test_file_path)
        test_df.loc[:, "dummy_labels"] = 0
        return test_df

    else:
        raise ValueError(f"mode {mode} is not valid")


def encode_ids(
    df: pd.DataFrame, config: DictConfig, le_encoder: LabelEncoder, save: bool = False
) -> pd.DataFrame:
    df.loc[:, "individual_id"] = le_encoder.fit_transform(df["individual_id"])
    if save:
        pickle_data_path = Path(config.output_path) / "le.pkl"
        with pickle_data_path.open("wb") as f:
            joblib.dump(le_encoder, f)
    return df


def create_folds(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    skf = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )
    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df["individual_id"])):
        df.loc[val_idx, "fold"] = fold
    df.loc[:, "fold"] = df["fold"].astype(int)
    logger.info(f"\n {df.fold.value_counts()}")
    return df


def preprocess_df(
    config: DictConfig,
    le_encoder: LabelEncoder,
    recache: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    train_df_path = (
        Path(config.data_path)
        / f"backfin-train-df-fold{config.n_splits}-seed{config.seed}.csv"
    )
    if not recache and train_df_path.exists():
        logger.info(f"cached file is loaded : {train_df_path}")
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
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        ),
        "val": A.Compose(
            [
                A.Resize(config.img_size, config.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        ),
        "test": A.Compose(
            [
                A.Resize(config["img_size"], config["img_size"]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        ),
    }
    return transform


# ==================
# infer funcs
# ==================
def get_predictions(
    test_df: pd.DataFrame, threshold: float = 0.2
) -> Dict[str, List[str]]:
    predictions: Dict[str, List[str]] = {}
    images = test_df["image"]
    targets = test_df["label"]
    distances = test_df["distances"]
    for image, target, distance in zip(images, targets, distances):
        if image in predictions:
            if len(predictions[image]) < 5:
                predictions[image].append(target)
        elif distance > threshold:
            predictions[image] = [target, "new_individual"]

        else:
            predictions[image] = ["new_individual", target]

    # validation + post process
    sample_list = [
        "938b7e931166",
        "5bf17305f073",
        "7593d2aee842",
        "7362d7a01d00",
        "956562ff2888",
    ]
    for key in predictions:
        if len(predictions[key]) < 5:
            remaining = [y for y in sample_list if y not in predictions]
            tmp_preds = predictions[key] + remaining
            predictions[key] = tmp_preds[:5]

    return predictions


# Ref
# https://www.kaggle.com/clemchris/pytorch-lightning-arcface-train-infer/notebook
def load_module(checkpoint_path: str, device: torch.device, config: DictConfig):
    dtype = torch.float if config.trainer.precision == 32 else torch.half
    module = MyLitModel.load_from_checkpoint(
        checkpoint_path,
        config=config,
        batch_size=config.batch_size,
        embedding_size=config.model.embedding_size,
        learning_rate=config.optimizer.default_lr,
    )
    module.to(device, dtype=dtype)
    module.eval()
    return module


def load_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: DictConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    datamodule = MyLitDataModule(train_df, val_df, test_df, config.batch_size, config)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    return train_loader, val_loader, test_loader


def load_encoder(le_path: str) -> LabelEncoder:
    with open(le_path, "rb") as f:
        le = joblib.load(f)
    return le


@torch.inference_mode()
def get_embeddings(
    module: pl.LightningModule,
    dataloader: DataLoader,
    encoder: LabelEncoder,
    config: DictConfig,
):
    all_img_names = []
    all_embeddings = []
    all_labels = []
    for idx, batch in enumerate(tqdm(dataloader)):
        if config.debug and idx == 50:
            break
        img_name = batch["image_path"]
        imgs = batch["image"].to(module.device, dtype=torch.half)
        labels = batch["label"].to(module.device, dtype=torch.long)

        _, emb = module(imgs, labels)

        all_img_names.append(img_name)
        emb = emb.cpu().numpy()

        # NaNの値があるとvstackのときにエラーが起きるからとりあえず0でreplaceする
        np.nan_to_num(emb, copy=False)
        all_embeddings.append(emb.astype("f4"))
        all_labels.append(labels.cpu().numpy())

    all_img_names = np.concatenate(all_img_names)
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels).astype("i4")

    all_embeddings = normalize(all_embeddings, axis=1, norm="l2").astype("f4")
    all_labels = encoder.inverse_transform(all_labels)
    return all_img_names, all_embeddings, all_labels


def create_val_targets_df(
    train_targets: np.ndarray, val_image_names: np.ndarray, val_targets: np.ndarray
) -> pd.DataFrame:

    allowed_targets = np.unique(train_targets)
    val_targets_df = pd.DataFrame(
        np.stack([val_image_names, val_targets], axis=1), columns=["image", "label"]
    )
    val_targets_df.loc[
        ~val_targets_df.label.isin(allowed_targets), "label"
    ] = "new_individual"

    return val_targets_df


def create_and_search_index(
    embedding_size: int,
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    index = faiss.IndexFlatIP(embedding_size)
    index.add(train_embeddings.astype("f4"))
    D, I = index.search(val_embeddings, k=k)
    return D, I


def create_distances_df(
    image_names: np.ndarray, labels: np.ndarray, D: np.ndarray, I: np.ndarray
) -> pd.DataFrame:
    def _my_code(
        image_names: np.ndarray, labels: np.ndarray, D: np.ndarray, I: np.ndarray
    ) -> pd.DataFrame:
        distance_df = []
        for i, image_name in enumerate(image_names):
            label = labels[I[i]]
            distances = D[i]
            subset_preds = {"label": label, "distances": distances, "image": image_name}
            distance_df.append(subset_preds)

        distance_df = pd.DataFrame(distance_df).reset_index(drop=True)
        logger.info(f"distance_df.head() = \n {distance_df.head()}")
        distance_df = (
            distance_df.groupby(["image", "label"])
            .distnaces.max()
            .reset_index(drop=True)
        )
        distance_df = distance_df.sort_values("distances", ascending=False).reset_index(
            drop=True
        )
        return distance_df

    def _ref_code(
        image_names: np.ndarray, labels: np.ndarray, D: np.ndarray, I: np.ndarray
    ) -> pd.DataFrame:
        distances_df = []
        for i, image_name in tqdm(enumerate(image_names)):
            label = labels[I[i]]
            distances = D[i]
            subset_preds = pd.DataFrame(
                np.stack([label, distances], axis=1), columns=["label", "distances"]
            )
            subset_preds["image"] = image_name
            distances_df.append(subset_preds)

        distances_df = pd.concat(distances_df).reset_index(drop=True)
        distances_df = (
            distances_df.groupby(["image", "label"]).distances.max().reset_index()
        )
        distances_df = distances_df.sort_values(
            "distances", ascending=False
        ).reset_index(drop=True)

        return distances_df

    return _ref_code(image_names, labels, D, I)


def get_best_threshold(
    val_targets_df: pd.DataFrame, val_df: pd.DataFrame
) -> Tuple[float, float]:
    best_thr = 0
    best_cv = 0
    for thr in np.arange(0.1, 0.9, 0.1):
        all_preds = get_predictions(val_df, threshold=thr)

        # cv = map_per_set(val_targets_df["label"], all_preds)
        cv = 0
        labels = val_targets_df["label"]
        images = val_targets_df["image"]
        for i, (label, image) in enumerate(zip(labels, images)):
            preds = all_preds[image]
            val_targets_df.loc[i, thr] = map_per_image(label, preds)

        logger.info(f"thr={thr}, cv={cv}")
        if cv > best_cv:
            best_cv = cv
            best_thr = thr

    logger.info(f"######### best_thr={best_thr}, best_cv={best_cv} ########## ")

    # Adjustment: Since Public lb has nearly 10% 'new_individual' (Be Careful for private LB)
    val_targets_df.loc[:, "is_new_individual"] = (
        val_targets_df["label"] == "new_individual"
    )
    logger.info(f"val_target_df.head() = \n {val_targets_df.head()}")
    val_scores = val_targets_df.groupby("is_new_individual").mean().T
    val_scores.loc[:, "adjusted_cv"] = val_scores[True] * 0.1 + val_scores[False] * 0.9
    best_thr = val_scores["adjusted_cv"].idxmax()
    logger.info(f"best_thr_adjusted={best_thr}")
    return best_thr, best_cv


def create_predictions_df(test_df: pd.DataFrame, best_thr: float) -> pd.DataFrame:
    predictions = get_predictions(test_df, best_thr)
    predictions = pd.Series(predictions).reset_index()
    predictions.columns = ["image", "predictions"]
    predictions.loc[:, "predictions"] = predictions["predictions"].map(
        lambda x: " ".join(x)
    )
    return predictions


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
        self._images = df["image"].values
        self._file_names = df["file_path"].values
        if mode == "test":
            self._labels = df["dummy_labels"].values
        else:
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
        org_img_path = self._images[idx]
        img = self.__get_img(img_path)
        label = self._labels[idx]

        if self._transform:
            img = self._transform[self._mode](image=img)["image"]

        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)

        return {
            "id": idx,
            "image_path": org_img_path,
            "image": img.to(dtype=self._dtype),
            "label": torch.tensor(label, dtype=self._dtype),
        }


class MyLitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        batch_size: int,
        config: DictConfig,
    ) -> None:
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._test_df = test_df
        self._config = config

        self.save_hyperparameters()

    def __create_dataset(self, mode: str) -> Dataset:
        if mode == "train":
            return HappyWhaleDataset(
                df=self._train_df, transform=get_transform(self._config), mode="train"
            )
        elif mode == "val":
            return HappyWhaleDataset(
                df=self._val_df, transform=get_transform(self._config), mode="val"
            )
        elif mode == "test":
            return HappyWhaleDataset(
                df=self._test_df, transform=get_transform(self._config), mode="test"
            )
        else:
            raise ValueError

    def train_dataloader(self) -> DataLoader:
        dataset = self.__create_dataset(mode="train")
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, **self._config.train_loader
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self.__create_dataset(mode="val")
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, **self._config.val_loader
        )

    def test_dataloader(self) -> DataLoader:
        dataset = self.__create_dataset(mode="test")
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, **self._config.test_loader
        )


# =============================
# Model
# =============================
class GeM(nn.Module):
    """GeM Pooling

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

    def gem(
        self, x: torch.Tensor, p: nn.parameter.Parameter, eps: float = 1e-6
    ) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + f"p={self._p.data.tolist()[0]:.4f}"
            + f",eps={self._eps})"
        )


class ArcMarginProduct(nn.Module):
    """Implement of large margin arc distance

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
        ls_eps: float = 0.0,
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
    def __init__(
        self,
        model_name: str,
        config: DictConfig,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pooling = GeM()
        self.drop = nn.Dropout(p=0.2, inplace=False)
        self.fc = nn.Linear(in_features, config.model.embedding_size)
        self.arc = ArcMarginProduct(
            512,
            config.num_classes,
            config=config,
            s=config.model.arc_params.s,
            m=config.model.arc_params.m,
            easy_margin=config.model.arc_params.easy_margin,
            ls_eps=config.model.arc_params.ls_eps,
        )

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        pooled_features = self.pooling(features).flatten(1)
        pooled_drop = self.drop(pooled_features)
        emb = self.fc(pooled_drop)
        output = self.arc(emb, labels)
        return output, emb


# TODO: 初期化にどんなパラメータが必要かが不明瞭になるからリファクタリングしたほうが良い
class MyLitModel(pl.LightningModule):
    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        embedding_size: int,
        config: DictConfig,
    ) -> None:
        super().__init__()
        self._config = config
        self.embedding_size = config.model.embedding_size
        self._criterion = eval(config.loss)()
        self._transform = get_transform(config)
        self._dtype = torch.float if config.trainer.precision == 32 else torch.half
        self.__build_model()
        self.save_hyperparameters()

        # settings of wandb logger
        self.monitor_metric = config.callbacks.monitor_metric
        self.monitor_mode = config.callbacks.mode

        # self._le = self.__load_le()
        self._test_preds_df_path = config.data_path + "/test/test.csv"

    def __build_model(self) -> None:
        self.model = HappyWhaleModel(
            model_name=self._config.model.name,
            config=self._config,
            pretrained=self._config.model.pretrained,
        )

    def __load_le(self) -> LabelEncoder:
        with open(self._config.output_path + "/le.pkl", "rb") as f:
            le = joblib.load(f)
        return le

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(images, labels)

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        if self.global_step == 0:
            self.logger.experiment.define_metric(self.monitor_metric, summary=self.monitor_mode)


        loss, pred, labels, img_path, idx, emb = self.__share_step(batch, "train")
        return {
            "loss": loss,
            "pred": pred,
            "labels": labels,
            "image_path": img_path,
            "ids": idx,
            "emb": emb,
        }

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        loss, pred, labels, img_path, idx, emb = self.__share_step(batch, "val")
        return {
            "pred": pred,
            "labels": labels,
            "image_path": img_path,
            "ids": idx,
            "emb": emb,
        }

    def __share_step(self, batch: dict, mode: str):
        images, labels = batch["image"].to(dtype=self._dtype), batch["label"].to(
            dtype=torch.long
        )
        img_path, idx = batch["image_path"], batch["id"]
        if mode == "train":
            outputs, emb = self.forward(images, labels)
        else:
            with torch.no_grad():
                outputs, emb = self.forward(images, labels)
        outputs = outputs.to(dtype=self._dtype)
        # print(f"output: {outputs.dtype}, labels: {labels.dtype}")
        # print(f"output: {outputs}, labels: {labels}")
        loss = self._criterion(outputs, labels)

        pred = outputs.sigmoid().detach().cpu()
        emb = emb.detach().cpu()
        labels = labels.detach().cpu()
        return loss, pred, labels, img_path, idx, emb

    def training_epoch_end(self, outputs: dict) -> None:
        self.__share_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: dict) -> None:
        self.__share_epoch_end(outputs, "val")

    def __share_epoch_end(self, outputs: dict, mode: Literal["train", "val"]) -> None:
        preds = []
        labels = []
        for out in outputs:
            pred, label = out["pred"], out["labels"]
            preds.append(pred)
            labels.append(label)

        preds = torch.cat(preds)
        labels = torch.cat(labels)
        loss = F.cross_entropy(preds.float(), labels.long())
        if mode == "train":
            self.log(f"fold{config.fold}_train_loss", loss)
        elif mode == "val":
            self.log(self._config.callbacks.monitor_metric, loss)

    def test_step_end(self, outputs):
        # TODO: 予測を本来のラベルにもどす
        preds = np.concatenate([out["pred"] for out in outputs])
        image_paths = np.concatenate([out["image_path"] for out in outputs])
        embs = np.vstack([out["emb"] for out in outputs])
        embs = normalize(embs, axis=1, norm="l2")

        df = pd.DataFrame({"predictions": preds, "image": image_paths})
        df.loc[:, "predictions"] = df["predictions"].map(lambda x: " ".join(x))
        df.to_csv(self._test_preds_df_path, index=False)

    # TODO: learning_rateもhparam経由に書き直す
    def configure_optimizers(self):
        optimizer = eval(self._config.optimizer.name)(
            self.parameters(),
            self.hparams.learning_rate,
            **self._config.optimizer.params,
        )
        scheduler = eval(self._config.scheduler.name)(
            optimizer, **self._config.scheduler.params
        )
        return [optimizer], [scheduler]


def train(model, datamodule, fold: int, config: DictConfig) -> None:

    # monitor metric
    config["callbacks"]["monitor_metric"] = (
        f"fold{fold}_" + config["callbacks"]["monitor_metric"]
    )
    config["fold"] = fold

    # instanciate callbacks
    earystopping = EarlyStopping(
        monitor=config.callbacks.monitor_metric,
        patience=config.callbacks.patience,
        verbose=True,
        mode=config.callbacks.mode,
    )
    lr_monitor = callbacks.LearningRateMonitor()
    loss_checkpoint = callbacks.ModelCheckpoint(
        dirpath=f"./output/{config.model.name}",
        filename=f"{config.model.name}"
        + f"-fold{config.n_splits}-{fold}"
        + "-{epoch}-{step}",
        monitor=config.callbacks.monitor_metric,
        save_top_k=1,
        mode=config.callbacks.mode,
        save_weights_only=False,
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
        callbacks=[lr_monitor, loss_checkpoint, earystopping],
        **config.trainer,
    )
    trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)


def infer(
    checkpoint_path: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    img_size: Tuple[int, int],
    batch_size: int,
    config: DictConfig,
    fold: int,
    k: int = 50,
):
    module = load_module(checkpoint_path, torch.device("cuda"), config)

    train_loader, val_loader, test_loader = load_dataloaders(
        train_df, val_df, test_df, config
    )

    encoder = load_encoder(config.output_path + "/le.pkl")

    train_img_names, train_embs, train_labels = get_embeddings(
        module,
        train_loader,
        encoder,
        config,
    )
    val_img_name, val_embs, val_labels = get_embeddings(
        module, val_loader, encoder, config
    )
    test_img_name, test_embs, test_labels = get_embeddings(
        module, test_loader, encoder, config
    )

    D, I = create_and_search_index(module.embedding_size, train_embs, val_embs, k)

    val_targets_df = create_val_targets_df(train_labels, val_img_name, val_labels)
    val_df = create_distances_df(val_img_name, train_labels, D, I)
    best_thr, best_cv = get_best_threshold(val_targets_df, val_df)

    train_embs = np.concatenate([train_embs, val_embs])
    train_labels = np.concatenate([train_labels, val_labels])

    D, I = create_and_search_index(module.embedding_size, train_embs, test_embs, k)
    test_df = create_distances_df(test_img_name, train_labels, D, I)
    predictions = create_predictions_df(test_df, best_thr)

    logger.info(f"prediciotns_df \n{predictions.head()}")

    # make submission
    submission_csv_path = (
        Path(config.output_path)
        / str(Path(__file__).stem)
        / f"fold{fold}-submission.csv"
    )
    submission_csv_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(submission_csv_path, index=False)


def update_config(config: DictConfig) -> DictConfig:

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fold", default=-1, type=int, nargs="*")
    parser.add_argument("--tpu")
    parser.add_argument("--tpu_cores", default=-10, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--exp", action="store_true")
    args = parser.parse_args()

    if args.train_fold != -1:
        config["train_fold"] = args.train_fold
        logger.info(f"train_epoch is specified: {config.train_fold}")

    if args.tpu:
        config["trainer"]["tpu_cores"] = args.tpu_cores
        config["trainer"].pop("gpus")
        logger.info(f"tpu is specified with the number of {config.trainer.tpu_cores}")

    if args.exp:
        logger.info(" ####### exp mode is called. ####### ")
        config["trainer"]["limit_train_batches"] = 0.5
        config["trainer"]["limit_val_batches"] = 0.5

    if args.debug:
        logger.info(" ####### debug mode is called. ####### ")
        config["trainer"]["limit_train_batches"] = 0.005
        config["trainer"]["limit_val_batches"] = 0.02
        config["trainer"]["max_epochs"] = 1
        config["debug"] = True

    return config


def main(config: DictConfig) -> None:
    config = update_config(config)
    pprint.pprint(config)

    for fold in range(config.n_splits):
        torch.autograd.set_detect_anomaly(True)
        seed_everything(config.seed)
        if (config.debug or config.trainer.limit_train_batches != 1.0) and fold > 0:
            break
        # prepare data
        le_encoder = LabelEncoder()
        df = preprocess_df(config, le_encoder)

        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]
        test_df = get_df(config, mode="test")

        plot_dist(
            label=val_df["individual_id"].values,
            save_name=config.output_path + f"/fold{fold}-dist.png",
        )

        datamodule = MyLitDataModule(
            train_df, val_df, test_df, config.batch_size, config
        )
        model = MyLitModel(
            config.batch_size,
            config.optimizer.default_lr,
            config.model.embedding_size,
            config,
        )

        if config.train and fold in config.train_fold:
            logger.info("#" * 8 + f"  Fold: {fold}  " + "#" * 8)
            train(model, datamodule, fold, config)

        if config.inference:
            # inference関数を実装する
            checkpoint_path = list(
                (Path(config.output_path) / config.model.name).glob(
                    f"{config.model.name}-fold{config.n_splits}-{fold}*"
                )
            )
            pprint.pprint(checkpoint_path)

            infer(
                checkpoint_path[0],
                train_df,
                val_df,
                test_df,
                config.img_size,
                config.batch_size,
                config,
                fold,
                50,
            )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(config)
