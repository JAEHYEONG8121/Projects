import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

        axes[i, 0].imshow(x[0].permute(1, 2, 0).numpy())
        axes[i, 0].set_title('Input Image')

        axes[i, 1].imshow(y[0].numpy(), cmap=cmap)
        axes[i, 1].set_title('Ground Truth')

        axes[i, 2].imshow(preds.numpy(), cmap=cmap)
        axes[i, 2].set_title('Prediction')

    plt.show()
