import argparse
import hashlib
import math
import os
import pprint
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import wandb
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

from PIL import Image

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
        exp_desc="exp002-batch512-emb1024-convnext-large-in22k-large-backfin-aug-v2-newlabel",
        train=False,
        debug=False,
        train_fold=[0, 1, 2, 3, 4],
        inference=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=2022,
        img_size=384,
        n_splits=5,
        num_classes=15587,
        data_path="./input/background-removed-happywhale-dataset",
        output_path="./output",
        log_path="wandb_logs",
        warm_start_path=None,
        wandb_project="HappyWhale",
        model=dict(
            # name="tf_efficientnet_b7_ns",
            # name="efficientnet_b6",
            # name="convnext_small",
            # name="convnext_large",
            name="convnext_large_in22ft1k",
            # name="swin_large_patch4_window7_224",
            # name="swin_base_patch4_window12_384_in22k",
            # name="beit_base_patch16_224_in22k",
            pretrained=True,
            embedding_size=512,
            arc_params=dict(s=10.0, m=0.50, ls_eps=0.0, easy_margin=False),
        ),
        trainer=dict(
            gpus=1,
            accumulate_grad_batches=8,
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
            benchmark=True,
            weights_summary="top",
            reload_dataloaders_every_epoch=True,
            auto_scale_batch_size="binsearch",
            auto_lr_find=False,
            max_epochs=30,
            stochastic_weight_avg=False,
            gradient_clip_val=0.5,
            # amp_backend="native",
            # amp_level="02",
        ),
        batch_size=28,
        train_loader=dict(shuffle=True, num_workers=4, pin_memory=True, drop_last=True),
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
        optimizer=dict(name="optim.AdamW", default_lr=3e-4, params=dict(weight_decay=1e-5)),
        # optimizer=dict(name="optim.Adam", default_lr=1e-2, params=dict(weight_decay=1e-6)),
        scheduler=dict(
            # name="optim.lr_scheduler.CosineAnnealingWarmRestarts",
            # params=dict(T_0=5, eta_min=1e-4),
            # name="optim.lr_scheduler.OneCycleLR",
            name="optim.lr_scheduler.CosineAnnealingLR",
            params=dict(T_max=30),
        ),
        loss="nn.CrossEntropyLoss",
        callbacks=dict(
            monitor_metric="val_loss",
            mode="min",
            patience=10,
            # patience=5,
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
    return np.mean([map_per_image(label, pred) for label, pred in zip(labels, predictions)])


def plot_dist(label: np.ndarray, save_name: str, figsize: Tuple[int, int] = (400, 200)) -> None:
    plt.figure(figsize=figsize)
    plt.hist(label, label="label")
    plt.savefig(save_name)


def get_df(config: DictConfig, mode: str = "train") -> pd.DataFrame:
    assert mode in {"train", "test"}
    if mode == "train":
        train_img_dir = Path(config.data_path) / "train_images"

        def _get_train_file_path(idx: str) -> Path:
            return train_img_dir / idx

        logger.info(f"train_img_dir = {train_img_dir}")

        train_df = pd.read_csv(Path(config.data_path) / "train.csv")
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
        logger.info(f"train_df.individual_id.nunique() = {train_df.individual_id.nunique()}")
        return train_df

    elif mode == "test":
        test_img_dir = Path(config.data_path) / "test_images"

        def _get_test_file_path(idx: str) -> Path:
            return test_img_dir / idx

        # test_df = pd.read_csv(Path(config.data_path) / "sample_submission.csv")
        test_df = pd.read_csv(Path(config.data_path) / "test.csv")
        test_df.loc[:, "file_path"] = test_df["image"].map(_get_test_file_path)
        test_df.loc[:, "dummy_labels"] = 0
        return test_df

    else:
        raise ValueError(f"mode {mode} is not valid")


def encode_ids(df: pd.DataFrame, config: DictConfig, le_encoder: LabelEncoder, save: bool = False) -> pd.DataFrame:
    df.loc[:, "individual_id"] = le_encoder.fit_transform(df["individual_id"])
    if save:
        pickle_data_path = Path(config.output_path) / "backfin-le"
        logger.info(f"saved label encocder: {pickle_data_path}")
        # classes = np.concatenate([np.array(["new_individual"]), le_encoder.classes_])
        # classes = np.append(le_encoder.classes_, "new_individual")
        classes = le_encoder.classes_
        np.save(pickle_data_path, classes)
        logger.info(f"claases: {len(classes)}")

        # with pickle_data_path.open("wb") as f:
        # joblib.dump(le_encoder, f)
    return df


def create_folds(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df["individual_id"])):
        df.loc[val_idx, "fold"] = fold
    df.loc[:, "fold"] = df["fold"].astype(int)
    logger.info(f"\n {df.fold.value_counts()}")
    return df


def preprocess_df(
    config: DictConfig,
    le_encoder: LabelEncoder,
    save: bool = True,
) -> pd.DataFrame:
    df = get_df(config)
    df = encode_ids(df, config, le_encoder, save)
    df = create_folds(df, config)
    return df


def get_transform(config: DictConfig) -> dict:
    transform = {
        "train": A.Compose(
            [
                A.Resize(config.img_size, config.img_size, interpolation=cv2.INTER_CUBIC),
                # A.OneOf(
                # [
                # A.augmentations.crops.transforms.CenterCrop(config.img_size, config.img_size),
                # A.augmentations.crops.transforms.RandomResizedCrop(config.img_size, config.img_size),
                # ],
                # p=0.5,
                # ),
                # A.augmentations.geometric.rotate.Rotate(limit=30, interpolation=cv2.INTER_CUBIC, p=0.5),
                # A.Rotate(limit=10, border_mode=cv2.BORDER_REPLICATE, p=0.5),
                A.Cutout(num_holes=8, max_h_size=2, max_w_size=2, fill_value=0, p=0.5),
                A.Cutout(num_holes=8, max_h_size=1, max_w_size=1, fill_value=1, p=0.5),
                A.HorizontalFlip(p=0.5),
                # A.OneOf(
                # [A.augmentations.transforms.Blur(p=0.3), A.augmentations.transforms.MotionBlur(p=0.3)], p=0.3
                # ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                # A.augmentations.transforms.GaussNoise(p=0.5),
                ToTensorV2(),
            ],
            p=1.0,
        ),
        "val": A.Compose(
            [
                A.Resize(config.img_size, config.img_size, interpolation=cv2.INTER_CUBIC),
                # A.augmentations.crops.transforms.CenterCrop(config.img_size, config.img_size),
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
                A.Resize(config.img_size, config.img_size, interpolation=cv2.INTER_CUBIC),
                # A.augmentations.crops.transforms.CenterCrop(config.img_size, config.img_size),
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
def get_predictions(test_df: pd.DataFrame, threshold: float = 0.2) -> Dict[str, List[str]]:
    logger.info(f"test_df.head() = \n {test_df.head()}")
    sample_list = ["938b7e931166", "5bf17305f073", "7593d2aee842", "7362d7a01d00", "956562ff2888"]
    images = test_df["image"].to_numpy()
    labels = test_df["label"].to_numpy()
    distances = test_df["distances"].to_numpy()

    predictions: Dict[str, List[str]] = {}
    for image, label, distance in tqdm(zip(images, labels, distances), total=len(test_df)):
        if image in predictions:
            if len(predictions[image]) == 5:
                continue
            predictions[image].append(label)
        elif distance > threshold:
            predictions[image] = [label, "new_individual"]

        else:
            predictions[image] = ["new_individual", label]

    # validation + post process
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
    logger.info(f"label encoder path = {le_path}")
    # with open(le_path, "rb") as f:
    # le = joblib.load(f)
    le = LabelEncoder()
    le.classes_ = np.load(le_path, allow_pickle=True)
    return le


def visualize_aug_image(dataloader, config, mode):
    import copy

    dataset = copy.deepcopy(dataloader.dataset)
    dataset._transform = {
        mode: A.Compose([p for p in dataset._transform[mode] if not isinstance(p, (A.Normalize, ToTensorV2))])
    }
    fig, ax = plt.subplots(5, 5, figsize=(200, 100))
    for i in range(5):
        for j in range(5):
            b = dataset[i]
            img = np.asarray(b["image"])
            img = img.astype("i4")
            ax[i, j].imshow(img)
            ax[i, j].set_title(str(Path(b["image_path"]).name))
            ax[i, j].set_axis_off()

    fig_path = Path(config.output_path) / Path(__file__).stem / f"{mode}-debug-img.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(fig_path))


@torch.inference_mode()
def get_embeddings(
    module: pl.LightningModule,
    dataloader: DataLoader,
    encoder: LabelEncoder,
    config: DictConfig,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_img_names = []
    all_embeddings = []
    all_labels = []

    for idx, batch in enumerate(tqdm(dataloader, desc=f"Creating {mode} embeddings")):
        img_name = batch["image_path"]
        imgs = batch["image"].to(module.device, dtype=torch.half)
        labels = batch["label"].to(module.device, dtype=torch.long)

        _, emb = module(imgs, labels)

        all_img_names.append(img_name)
        emb = emb.cpu().numpy()

        # NaN??????????????????vstack??????????????????????????????????????????????????????0???replace??????
        np.nan_to_num(emb, copy=False)
        all_embeddings.append(emb)
        all_labels.append(labels.cpu().numpy())

    all_img_names = np.concatenate(all_img_names)
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)

    all_embeddings = normalize(all_embeddings, axis=1, norm="l2")
    all_labels = encoder.inverse_transform(all_labels)
    return all_img_names, all_embeddings, all_labels


def create_val_targets_df(
    train_targets: np.ndarray, val_image_names: np.ndarray, val_targets: np.ndarray
) -> pd.DataFrame:

    allowed_targets = np.unique(train_targets)
    val_targets_df = pd.DataFrame(np.stack([val_image_names, val_targets], axis=1), columns=["image", "label"])
    val_targets_df.loc[~val_targets_df.label.isin(allowed_targets), "label"] = "new_individual"

    return val_targets_df


def create_and_search_index(
    embedding_size: int, train_embeddings: np.ndarray, val_embeddings: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    index = faiss.IndexFlatIP(embedding_size)
    index.add(train_embeddings.astype("f4"))
    D, I = index.search(val_embeddings.astype("f4"), k=k)
    return D, I


def create_distances_df(image_names: np.ndarray, labels: np.ndarray, D: np.ndarray, I: np.ndarray) -> pd.DataFrame:
    def _my_code(image_names: np.ndarray, labels: np.ndarray, D: np.ndarray, I: np.ndarray) -> pd.DataFrame:
        distance_df = []
        for i, image_name in enumerate(image_names):
            label = labels[I[i]]
            distances = D[i]
            subset_preds = {"label": label, "distances": distances, "image": image_name}
            distance_df.append(subset_preds)

        distance_df = pd.DataFrame(distance_df).reset_index(drop=True)
        distance_df = distance_df.groupby(["image", "label"]).distnaces.max().reset_index(drop=True)
        distance_df = distance_df.sort_values("distances", ascending=False).reset_index(drop=True)
        logger.info(f"distance_df.head() = \n {distance_df.head()}")
        return distance_df

    def _ref_code(image_names: np.ndarray, labels: np.ndarray, D: np.ndarray, I: np.ndarray) -> pd.DataFrame:
        distances_df = []
        for i, image_name in tqdm(enumerate(image_names)):
            label = labels[I[i]]
            distances = D[i]
            subset_preds = pd.DataFrame(np.stack([label, distances], axis=1), columns=["label", "distances"])
            subset_preds["image"] = image_name
            distances_df.append(subset_preds)

        distances_df = pd.concat(distances_df).reset_index(drop=True)
        distances_df = distances_df.groupby(["image", "label"]).distances.max().reset_index()
        distances_df = distances_df.sort_values("distances", ascending=False).reset_index(drop=True)
        logger.info(f"distance_df.head() = \n {distances_df.head()}")

        return distances_df

    return _ref_code(image_names, labels, D, I)


def get_best_threshold(val_targets_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[float, float]:
    labels = val_targets_df["label"]
    images = val_targets_df["image"]
    best_thr = 0
    best_cv = 0
    for thr in np.arange(0.1, 1.0 + 1e-6, 0.1):
        all_preds = get_predictions(val_df, threshold=thr)
        cv = 0
        for i, (label, image) in enumerate(zip(labels, images)):
            preds = all_preds[image]
            val_targets_df.loc[i, thr] = map_per_image(label=label, predictions=preds)

        cv = val_targets_df[thr].mean()
        logger.info(f" ##### thr={thr:.2f}, cv={cv} ##### ")
        if cv > best_cv:
            best_cv = cv
            best_thr = thr

    # Adjustment: Since Public lb has nearly 10% 'new_individual' (Be Careful for private LB)
    val_targets_df.loc[:, "is_new_individual"] = val_targets_df["label"] == "new_individual"
    val_scores = val_targets_df.groupby("is_new_individual").mean().T
    val_scores.loc[:, "adjusted_cv"] = val_scores[True] * 0.1 + val_scores[False] * 0.9
    best_thr = val_scores["adjusted_cv"].idxmax()

    logger.info(f"######### best_thr={best_thr:.2f}, best_cv={best_cv} ########## ")
    logger.info(f"val_target_df.head() = \n {val_targets_df.head()}")
    logger.info(f"val_scores.head() = {val_scores.head()}")
    logger.info(f"best_thr_adjusted = {best_thr}")

    return best_thr, best_cv


def create_predictions_df(test_df: pd.DataFrame, best_thr: float) -> pd.DataFrame:
    predictions = get_predictions(test_df, best_thr)
    predictions = pd.Series(predictions).reset_index()
    predictions.columns = ["image", "predictions"]
    predictions.loc[:, "predictions"] = predictions["predictions"].map(lambda x: " ".join(x))
    return predictions


# =============================
# Data
# =============================
class HappyWhaleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str, transform: Optional[dict] = None) -> None:
        assert mode in {"train", "val", "test"}
        super().__init__()
        self._df = df
        self._images = df["image"].to_numpy()
        self._file_names = df["file_path"].to_numpy()
        if mode == "test":
            self._labels = df["dummy_labels"].to_numpy()
        else:
            self._labels = df["individual_id"].to_numpy()

        self._mode = mode
        self._transform = transform

        self._horizontal_flip = A.Compose([A.HorizontalFlip(p=1.0)])

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

        # make new label (new_individual)
        # if torch.rand(1)[0] < 0.1:
        #     img = self._horizontal_flip(image=img)["image"]
        #     label = np.int32(15587 + 1)
        # label = np.zeros(15587)

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

        logger.info(f"batch_size = {self.hparams.batch_size}")
        logger.info(f"train_df.individual_id.nunique() = {self._train_df.individual_id.nunique()}")

    def __create_dataset(self, mode: str) -> Dataset:
        if mode == "train":
            return HappyWhaleDataset(df=self._train_df, transform=get_transform(self._config), mode="train")
        elif mode == "val":
            return HappyWhaleDataset(df=self._val_df, transform=get_transform(self._config), mode="val")
        elif mode == "test":
            return HappyWhaleDataset(df=self._test_df, transform=get_transform(self._config), mode="test")
        else:
            raise ValueError

    def train_dataloader(self) -> DataLoader:
        dataset = self.__create_dataset(mode="train")
        return DataLoader(dataset, batch_size=self.hparams.batch_size, **self._config.train_loader)

    def val_dataloader(self) -> DataLoader:
        dataset = self.__create_dataset(mode="val")
        return DataLoader(dataset, batch_size=self.hparams.batch_size, **self._config.val_loader)

    def test_dataloader(self) -> DataLoader:
        dataset = self.__create_dataset(mode="test")
        return DataLoader(dataset, batch_size=self.hparams.batch_size, **self._config.test_loader)


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
        assert x.ndim == 3 or x.ndim == 4, ValueError(
            f"x.ndim != 3 | x.ndm != 4 : x.ndim {x.ndim} & x.shape {x.shape}"
        )
        return self.gem(x, p=self._p, eps=self._eps)

    def gem(self, x: torch.Tensor, p: nn.parameter.Parameter, eps: float = 1e-6) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + f"p={self._p.data.tolist()[0]:.4f}" + f",eps={self._eps})"


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

        # self.norm = nn.BatchNorm1d(in_features)

        self._device = torch.device(config.device)
        self._dtype = torch.float if config.trainer.precision == 32 else torch.half

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # cos(theta) & phi(theta)
        # you should do batch norm operation before multiply by weights
        # Ref
        # * https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/109987
        # * https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/315129
        # x = self.norm(x)
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
        in_features = self.backbone.get_classifier().in_features
        # self.backbone.classifier = nn.Identity()
        # self.backbone.global_pool = nn.Identity()
        self.drop = nn.Dropout(p=0.5, inplace=False)
        self.fc = nn.Linear(in_features, config.model.embedding_size)
        # self.fc = nn.LazyLinear(out_features=config.model.embedding_size)
        self.backbone.reset_classifier(num_classes=0, global_pool="avg")

        # self.pooling = GeM()
        self.arc = ArcMarginProduct(
            config.model.embedding_size,
            config.num_classes,
            config=config,
            s=config.model.arc_params.s,
            m=config.model.arc_params.m,
            easy_margin=config.model.arc_params.easy_margin,
            ls_eps=config.model.arc_params.ls_eps,
        )

        self.norm1 = nn.BatchNorm1d(in_features, affine=False)
        # self.norm1 = nn.LayerNorm(in_features)
        # self.norm2 = nn.BatchNorm1d(config.model.embedding_size)

    def freaze_layers(self):
        logger.info(" ##### freeze layers ##### ")
        for param in self.backbone.parameters():
            param.require_grad = False

    def unfreeze_layers(self):
        logger.info(" ##### unfreeze layers ##### ")
        for param in self.backbone.parameters():
            param.require_grad = True

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        # features = self.pooling(features).flatten(1)
        features = self.norm1(features)
        features = self.drop(features)
        emb = self.fc(features)
        # emb = self.norm2(emb)
        output = self.arc(emb, labels)
        return output, emb


# TODO: ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
class MyLitModel(pl.LightningModule):
    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        embedding_size: int,
        len_train_dl: int,
        config: DictConfig,
    ) -> None:
        super().__init__()
        self._config = config
        self.embedding_size = config.model.embedding_size
        # self._criterion = eval(config.loss)()
        self._criterion = F.cross_entropy
        self._transform = get_transform(config)
        self._dtype = torch.float if config.trainer.precision == 32 else torch.half
        self.__build_model()

        # settings of wandb logger
        self.monitor_metric = config.callbacks.monitor_metric
        self.monitor_mode = config.callbacks.mode
        self.logger_flag = True
        self.unfreeze_flag = True

        # self._le = self.__load_le()
        self._test_preds_df_path = config.data_path + "/test/test.csv"

        self.save_hyperparameters()

    def __build_model(self) -> None:
        self.model = HappyWhaleModel(
            model_name=self._config.model.name,
            config=self._config,
            pretrained=self._config.model.pretrained,
        )
        self.model.freaze_layers()

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(images, labels)

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        if self.global_step == 0 and self.logger_flag:
            self.logger.experiment.define_metric(self.monitor_metric, summary=self.monitor_mode)
            self.logger_flag = False

        if self.current_epoch > 5 and self.unfreeze_flag:
            self.model.unfreeze_layers()
            # self.hparams.learning_rate = self.hparams.learning_rate / 10
            self.unfreeze_flag = False

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
        images, labels = batch["image"].to(dtype=self._dtype), batch["label"].to(dtype=torch.long)
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
        metric = F.cross_entropy(preds.float(), labels.long())
        if mode == "train":
            self.log(f"fold{self._config.fold}_train_loss", metric)
        elif mode == "val":
            self.log(self._config.callbacks.monitor_metric, metric)

    def test_step_end(self, outputs):
        # TODO: ???????????????????????????????????????
        preds = np.concatenate([out["pred"] for out in outputs])
        image_paths = np.concatenate([out["image_path"] for out in outputs])
        embs = np.vstack([out["emb"] for out in outputs])
        embs = normalize(embs, axis=1, norm="l2")

        df = pd.DataFrame({"predictions": preds, "image": image_paths})
        df.loc[:, "predictions"] = df["predictions"].map(lambda x: " ".join(x))
        df.to_csv(self._test_preds_df_path, index=False)

    # TODO: learning_rate???hparam?????????????????????
    def configure_optimizers(self):
        optimizer = eval(self._config.optimizer.name)(
            self.parameters(),
            self.hparams.learning_rate,
            **self._config.optimizer.params,
        )
        if self._config.scheduler.name == "optim.lr_scheduler.OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.hparams.learning_rate,
                # steps_per_epoch=(self.hparams.len_train_dl//self.hparams.batch_size)//self.trainer.accumulate_grad_batches,
                steps_per_epoch=(
                    (self.hparams.len_train_dl // self.hparams.batch_size) // self.trainer.accumulate_grad_batches
                ),
                epochs=self.trainer.max_epochs,
                anneal_strategy="cos",
                div_factor=25,  # init_lr = max_lr / div_factor
                verbose=False,
            )
            scheduler = {"scheduler": scheduler, "interval": "step"}
        else:
            scheduler = eval(self._config.scheduler.name)(
                optimizer,
                **self._config.scheduler.params,
            )
        return [optimizer], [scheduler]


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold: int,
    config: DictConfig,
) -> None:

    # monitor metric
    config["callbacks"]["monitor_metric"] = f"fold{fold}_" + config["callbacks"]["monitor_metric"]
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
        dirpath=f"./output/checkpoints/{config.model.name}/{config.exp_desc}",
        filename=f"{config.model.name}" + f"-fold{config.n_splits}-{fold}" + "-{epoch}-{step}",
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

    model = MyLitModel(
        config.batch_size,
        config.optimizer.default_lr,
        config.model.embedding_size,
        len(train_df),
        config,
    )

    trainer = pl.Trainer(
        logger=pl_logger,
        callbacks=[lr_monitor, loss_checkpoint, earystopping],
        **config.trainer,
    )

    datamodule = MyLitDataModule(train_df, val_df, test_df, config.batch_size, config)

    if not config.debug:
        # trainer.tune(model, datamodule=datamodule)
        logger.info(f"overwrited batch_size = {model.hparams.batch_size}")

    trainer.fit(model, datamodule=datamodule)

    # remove fold{fold}_ prefix
    config["callbacks"]["monitor_metric"] = "_".join(config["callbacks"]["monitor_metric"].split("_")[1:])


def save_cv(cv: float, fold: int, cv_csv_path: Path):
    import json

    if not cv_csv_path.exists() or fold == 0:
        logger.info(f"make cv_csv :{cv_csv_path}")
        cv_dict = dict()
    else:
        try:
            logger.info(f"cv_csv exists so load : {cv_csv_path}")
            with cv_csv_path.open("r") as f:
                cv_dict = json.load(f)
        except:
            cv_dict = dict()

    cv_dict[f"fold{fold}-cv"] = cv
    logger.info(f"cv_csv = \n{cv_dict}")
    with cv_csv_path.open("w") as f:
        json.dump(cv_dict, f)


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

    logger.info(f"train_df.individual_id.nunique() = {train_df.individual_id.nunique()}")
    logger.info(f"val_df.individual_id.nunique() = {val_df.individual_id.nunique()}")
    train_loader, val_loader, test_loader = load_dataloaders(train_df, val_df, test_df, config)

    encoder = load_encoder(config.output_path + "/backfin-le.npy")
    logger.info(f"infer classes: {len(encoder.classes_)}")

    train_img_names, train_embs, train_labels = get_embeddings(module, train_loader, encoder, config, "train")
    val_img_name, val_embs, val_labels = get_embeddings(module, val_loader, encoder, config, "val")
    test_img_name, test_embs, test_labels = get_embeddings(module, test_loader, encoder, config, "test")

    D, I = create_and_search_index(module.embedding_size, train_embs, val_embs, k)
    logger.info(" Created index with train_embs")

    val_targets_df = create_val_targets_df(train_labels, val_img_name, val_labels)
    logger.info(f"val_targets_df.head() = \n {val_targets_df.head()}")

    val_df = create_distances_df(val_img_name, train_labels, D, I)
    best_thr, best_cv = get_best_threshold(val_targets_df, val_df)
    val_predictions = create_predictions_df(val_df, best_thr)
    val_df_path = Path(config.output_path) / str(Path(__file__).stem) / f"fold{fold}-val-df.csv"
    val_predictions.to_csv(val_df_path, index=False)
    save_cv(
        best_cv,
        fold,
        cv_csv_path=(Path(config.output_path) / (config.exp_desc + ".json")),
    )

    train_embs = np.concatenate([train_embs, val_embs])
    train_labels = np.concatenate([train_labels, val_labels])

    emb_path = Path(config.output_path) / str(Path(__file__).stem) / f"fold{fold}-embs"
    np.save(emb_path, train_embs)

    # seach index from test embeddings
    D, I = create_and_search_index(module.embedding_size, train_embs, test_embs, k)
    test_df = create_distances_df(test_img_name, train_labels, D, I)
    predictions = create_predictions_df(test_df, best_thr)
    logger.info(f"prediciotns_df \n{predictions.head()}")

    public_predicitons_path = Path("./input") / "0-720-eff-b5-640-rotate" / "submission.csv"
    public_predictions = pd.read_csv(public_predicitons_path)

    ids_without_backfin_path = Path("./input") / "ids-without-backfin" / "ids_without_backfin.npy"
    ids_without_backfin = np.load(ids_without_backfin_path, allow_pickle=True)

    ids2 = public_predictions["image"][~public_predictions["image"].isin(predictions["image"])]

    predictions = pd.concat(
        [
            predictions[~(predictions["image"].isin(ids_without_backfin))],
            public_predictions[public_predictions["image"].isin(ids_without_backfin)],
            public_predictions[public_predictions["image"].isin(ids2)],
        ]
    )
    predictions.drop_duplicates()

    # make submission
    submission_csv_path = Path(config.output_path) / str(Path(__file__).stem) / f"fold{fold}-submission.csv"
    submission_csv_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(submission_csv_path, index=False)
    logger.info(f"sub file path = {submission_csv_path}")


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
    pprint.pprint(dict(config))

    for fold in range(config.n_splits):
        torch.autograd.set_detect_anomaly(True)
        seed_everything(config.seed)
        if (config.debug or config.trainer.limit_train_batches != 1.0) and fold > 0:
            break
        # prepare data
        le_encoder = LabelEncoder()
        df = preprocess_df(config, le_encoder, save=True)

        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        test_df = get_df(config, mode="test")

        plot_dist(
            label=val_df["individual_id"].values,
            save_name=config.output_path + f"/fold{fold}-dist.png",
        )
        if config.debug:
            train_loader, val_loader, test_loader = load_dataloaders(train_df, val_df, test_df, config)
            visualize_aug_image(train_loader, config, "train")

        if config.train and fold in config.train_fold:
            logger.info("#" * 8 + f"  Fold: {fold}  " + "#" * 8)
            train(train_df, val_df, test_df, fold, config)

        if config.inference:
            # inference?????????????????????
            checkpoint_path = list(
                (Path(config.output_path) / "checkpoints" / config.model.name / config.exp_desc).glob(
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
                100,
            )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(config)
