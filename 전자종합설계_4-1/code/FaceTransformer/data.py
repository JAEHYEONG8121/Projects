import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

labels_dir = r"C:\Users\jaehy\FTproject\cv-rl\LAPA\LaPa\train\labels" # 레이블 디렉토리 경로를 여기에 입력하세요
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.png')]

for label_file in label_files[:5]:  # 처음 5개 파일만 확인
    label_path = os.path.join(labels_dir, label_file)
    label = Image.open(label_path).convert("L")
    label_array = np.array(label)
    print(f"File: {label_file}, Min: {label_array.min()}, Max: {label_array.max()}")

# 사용자 정의 컬러맵 생성 (0~10의 클래스 값을 컬러로 매핑)
cmap = ListedColormap(['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white', 'orange', 'purple', 'brown'])

# 처음 5개 파일 시각화
for label_file in label_files[:5]:
    label_path = os.path.join(labels_dir, label_file)
    label = Image.open(label_path).convert("L")
    label_array = np.array(label)

    plt.imshow(label_array, cmap=cmap)
    plt.colorbar(ticks=range(11))  # 클래스 값에 대한 색상 바
    plt.title(f"Label: {label_file}")
    plt.show()
