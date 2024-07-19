import torch
from pytorch_lightning import Trainer
from LAPA_dataloader import LAPASegmentationDataModule
from FT_train_segmentation_reinforce import SemanticSegmentationModel
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 체크포인트 파일 경로
checkpoint_path = r'C:\Users\jaehy\FTproject\cv-rl\LAPA\CheckPoint\checkpoint-epoch=49-val_loss=0.00.ckpt'

# 모델 로드
model = SemanticSegmentationModel.load_from_checkpoint(checkpoint_path)
model.eval()

# 데이터 로드
data_module = LAPASegmentationDataModule(data_dir=r'C:\Users\jaehy\FTproject\LAPA\LaPa', batch_size=32, image_size=256)
data_module.setup(stage='test')

# 시각화 함수
def visualize_predictions(model, datamodule, num_samples=5):
    datamodule.setup(stage='test')
    dataloader = datamodule.test_dataloader()
    
    samples = []
    for batch in dataloader:
        x, y = batch
        preds = model(x.to(model.device)).argmax(dim=1).cpu()
        samples.append((x, y, preds))
        if len(samples) >= num_samples:
            break

    cmap = ListedColormap(['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white', 'orange', 'purple', 'brown'])

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    for i, (x, y, preds) in enumerate(samples):
        axes[i, 0].imshow(x[0].permute(1, 2, 0).numpy())
        axes[i, 0].set_title('Input Image')
        axes[i, 1].imshow(y[0][0].numpy(), cmap=cmap)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 2].imshow(preds[0].numpy(), cmap=cmap)
        axes[i, 2].set_title('Prediction')

    plt.show()

# 모델 예측 및 시각화
visualize_predictions(model, data_module, num_samples=5)
