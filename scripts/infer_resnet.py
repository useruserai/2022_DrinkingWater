import torch
import yaml
import sys
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from DL_Lecture.models.resnet import ResNet32_model

def main():
    print('ResNet for CIFAR10 evaluation')

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/cifar10_resnet.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    # GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    timestamp = "1743655815"
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))

    # 데이터 로드 (단일 이미지 테스트용으로 수정)
    transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 단일 이미지 로드 (강아지 사진)
    img_path = 'C:/Users/Catholic/PycharmProjects/PythonProject/DL_Lecture/scripts/example/5/dog.jpg'
    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0)  # 모델 입력 형태에 맞게 변환

    # 모델 초기화
    model = ResNet32_model().to(device)  # 모델을 지정한 device로 올려줌, dropout x
    model.eval()

    # 저장된 state 불러오기
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints/best.pth"))
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 예측 수행
    with torch.no_grad():
        pred = model(img_tensor.to(device))
        _, top_pred = torch.topk(pred, k=1, dim=-1)
        top_pred = top_pred.squeeze(dim=1)

        # 정규화 되돌리기
        img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img = img * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])

        # 값 범위가 [0, 1]로 조정될 수 있도록 처리
        img = np.clip(img, 0, 1)

        # 이미지 출력
        plt.imshow(img)
        plt.show()

        print("--------------------------------------")
        print("truth: dog")  # 실제 레이블은 dog
        print("model prediction:", classes[top_pred])

if __name__ == "__main__":
    main()