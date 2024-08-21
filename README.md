# **CNN_classification**

------------------------

## **(예제) O, X, Δ 도형의 손그림 이미지를 CNN으로 학습을 시켜 데이터를 분류 하는 모델을 만들어보자.**

## **Dataset**
O, X, Δ 도형의 이미지 데이터셋 준비 및 모자란 그림은 그림판을 사용해 직접 드로잉 ( 각각 50개 )

## **Model Setting**

```
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(32, 64, kernel_size=3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(64, 64, kernel_size=3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
).to(device)
```

----------

## **평가**

Epoch 50 학습
Loss: 0.002929 | Accuracy: 83.33%

![image](https://github.com/Songysp/CNN/assets/156406181/cf66e529-8718-4e8d-ab13-44edb3e28d8b)
