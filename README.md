
- [MNIST](#mnist)
  - [개발 환경](#개발-환경)
      - [패키지 설치](#패키지-설치)
  - [주요 파일 설명](#주요-파일-설명)
  - [명령어](#명령어)
      - [mnist 이미지 미리보기](#mnist-이미지-미리보기)
      - [학습하기](#학습하기)
      - [테스트하기](#테스트하기)

---

## 개발 환경

python 3.8 이상

```
numpy==1.24.2
matplotlib==3.6.3
urllib3==1.26.14
opencv-python==4.7.0.68
```

#### 패키지 설치
```bash
$ pip install -r requirements.txt
```

## 주요 파일 설명

```
.
├── deep_convnet.py         # Deep Convolution Network
├── params.pkl              # 학습후 생성되는 weights 파일
├── predict_cam.py          # mnist 숫자 인식 테스트 (캠 사용)
├── train_convnet.py        # 학습하기
├── mnist_show.py           # mnist 시각
└── requirements.txt        # 코드를 구동하기 위해 필요한 모듈 리스트
```

## 명령어

#### mnist 이미지 미리보기

```bash
$ python mnist_show.py
```

#### 학습하기

```bash
$ python train_convnet.py
```

#### 테스트하기

```bash

$ python predict_cam.py               # mnist 숫자 인식 테스트 (캠 사용)
```
