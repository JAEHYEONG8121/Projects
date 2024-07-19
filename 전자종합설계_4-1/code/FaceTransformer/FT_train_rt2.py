from typing import Optional, Tuple
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torchmetrics import MetricCollection, Dice
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from LAPA_dataloader import LAPADataset, LAPASegmentationDataModule
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import random
import numpy as np
from torch.distributions import Bernoulli
from sklearn.metrics import f1_score

import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout, emb_dropout):
        super(ViT, self).__init__()
        self.conv = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_patch_embedding = nn.Conv2d(dim, dim, kernel_size=1)
        self.deconv = nn.ConvTranspose2d(dim, num_classes, kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)  # [batch_size, dim, h', w']
        x = x.flatten(2).transpose(1, 2)  # [batch_size, h'*w', dim]
        x = self.transformer(x)  # [batch_size, h'*w', dim]
        x = x.transpose(1, 2).view(x.size(0), -1, int(x.size(1)**0.5), int(x.size(1)**0.5))  # [batch_size, dim, h', w']
        x = self.to_patch_embedding(x)  # [batch_size, dim, h', w']
        x = self.deconv(x)  # [batch_size, num_classes, H, W]
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class SemanticSegmentationModel(LightningModule):
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 32,
        num_classes: int = 11,
        dim: int = 1024,
        depth: int = 6,
        heads: int = 16,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        optimizer: str = "adam",
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "none",
        warmup_steps: int = 0,
        weights: Optional[str] = None,
        prefix: str = "net",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.lr = lr
        self.betas = betas
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.weights = weights
        self.prefix = prefix    

        self.net = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout
        )

        if self.weights:
            state_dict = torch.load(self.weights)
            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]

            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith(self.prefix):
                    continue

                k = k.replace(self.prefix + ".", "")
                new_state_dict[k] = v

            self.net.load_state_dict(new_state_dict, strict=True)

    def setup(self, stage=None):
        device = self.device if self.device else 'cpu'
        self.train_metrics = MetricCollection([Dice(num_classes=self.hparams.num_classes)]).to(device)
        self.val_metrics = MetricCollection([Dice(num_classes=self.hparams.num_classes)]).to(device)
        self.test_metrics = MetricCollection([Dice(num_classes=self.hparams.num_classes)]).to(device)

    def forward(self, x):
        return self.net(x)

    def shared_step(self, batch, mode="train"):
        x, y = batch
        device = self.device if self.device else 'cpu'
        x, y = x.to(device), y.to(device)

        pred = self.forward(x)

        # 크기 맞추기: [N, num_classes, H, W] -> [N, H, W]
        pred = torch.argmax(pred, dim=1)

        # 크기 변환 확인
        y_flat = y.flatten()
        pred_flat = pred.flatten()

        # F1 점수 계산
        reward = f1_score(y_flat.cpu(), pred_flat.cpu(), average='micro')

        # Bernoulli 분포 생성
        dist = torch.distributions.Bernoulli(logits=pred.float())
        sample = dist.sample()
        log_prob = dist.log_prob(sample)

        # 로스 계산
        loss = -torch.mean(log_prob * reward)
        
        # Ensure loss requires gradient
        if not loss.requires_grad:
            loss.requires_grad = True

        # 메트릭 계산
        metrics = getattr(self, f"{mode}_metrics")(pred, y.long())

        self.log(f"{mode}_loss", loss, on_epoch=True, prog_bar=True)
        for k, v in metrics.items():
            self.log(f"{mode}_{k.lower()}", v, on_epoch=True)

        return loss




    def training_step(self, batch, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return self.shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self.shared_step(batch, "val")

    def test_step(self, batch, _):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        return optimizer


def visualize_predictions(model, datamodule, num_samples=5):
    datamodule.setup(stage='test')
    dataloader = datamodule.test_dataloader()
    
    all_samples = []
    for batch in dataloader:
        x, y = batch
        preds = model(x.to(model.device)).argmax(dim=1).cpu()
        all_samples.extend(list(zip(x, y, preds)))
    
    # 무작위로 샘플을 선택
    samples = random.sample(all_samples, num_samples)

    cmap = ListedColormap(['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white', 'orange', 'purple', 'brown'])

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    for i, (x, y, preds) in enumerate(samples):
        # 시각화 전 데이터 확인
        y_min, y_max = y.min(), y.max()
        preds_min, preds_max = preds.min(), preds.max()
        print(f"Sample {i}: y_min={y_min}, y_max={y_max}, preds_min={preds_min}, preds_max={preds_max}")

        # 이미지 데이터 클리핑 및 범위 조정
        #x = x[0].permute(1, 2, 0).numpy()
        #x = np.clip((x + 1) / 2, 0, 1)  # [-1, 1] 범위를 [0, 1]로 변환
        
        axes[i, 0].imshow(x.permute(1, 2, 0).numpy())
        axes[i, 0].set_title('Input Image')
        axes[i, 1].imshow(y[0].numpy(), cmap=cmap)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 2].imshow(preds.numpy(), cmap=cmap)
        axes[i, 2].set_title('Prediction')

    plt.show()

if __name__ == "__main__":
    # 데이터 디렉토리와 배치 크기를 설정합니다.
    data_dir = r"C:\Users\jaehy\FTproject\cv-rl\LAPA\LaPa"  # 실제 데이터셋 경로로 수정하세요.
    batch_size = 32
    image_size = 256  # 이미지 크기 설정
    
    # 데이터 모듈을 생성합니다.
    data_module = LAPASegmentationDataModule(data_dir=data_dir, batch_size=batch_size, image_size=image_size)
    
    # 체크포인트 경로 설정
    checkpoint_dir = r'C:\Users\jaehy\FTproject\cv-rl\LAPA\CheckPoint'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.ckpt')
    
    # 모델을 생성합니다.
    model = SemanticSegmentationModel(image_size=image_size, num_classes=11)  # LAPA 데이터셋 클래스 수에 맞게 설정

    # 로그를 설정합니다.
    logger = CSVLogger(save_dir="output", name="default")
    
    # 콜백을 설정합니다.
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,  # 모델이 저장될 경로
        filename='checkpoint-{epoch:02d}-{val_loss:.2f}',  # 저장될 파일명 형식
        save_top_k=1,  # 가장 좋은 모델 하나만 저장
        verbose=True,
        monitor='val_loss',  # val_loss 기준으로 저장
        mode='min'  # 가장 낮은 val_loss 기준으로 저장
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", 
        mode="min", 
        patience=10
    )
    
    # 트레이너를 생성합니다.
    trainer = Trainer(
        max_epochs=30,  # 에포크 수를 50으로 설정
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        check_val_every_n_epoch=10
    )
    
    # 학습을 시작합니다.
    trainer.fit(model, data_module)
    
    # 모델 평가
    trainer.test(model, datamodule=data_module)
    
    # 평가 후 시각화
    visualize_predictions(model, data_module)
