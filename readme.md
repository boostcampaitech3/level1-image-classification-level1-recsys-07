# Image Classification
### Folder Structure
```
├─assets
└─code
    ├─dataset
    │      dataset.py
	│			MaskBaseDataset
	│			MaskSplitByProfileDataset
	│			TestDataset
	│			SamplingMaskDataset
    ├─loss
    │      loss.py
	│			FocalLoss
	│			LabelSmoothingLoss
	│			F1Loss
	│			CrossEntropyLoss
    ├─model
    │      model.py
	│			resnet18
	│			vit_small_r26_s32_224_in21k
	│			efficientnetv2_rw_t
    └─train
            train.py
```
# 개요
## 프로젝트 주제
- 카메라로 촬영한 사람 얼굴 이미지의 마스크 착용 여부 / 나이 / 성별를 판단하는 Task
- 나이, 성별, 마스크 유무를 기준으로 18개의 라벨 분류하기
![enter image description here](https://github.com/boostcampaitech3/level1-image-classification-level1-recsys-07/blob/main/assets/1.png)
## 프로젝트 개요(프로젝트 구현 내용, 컨셉, 교육 내용과의 관련성 등)
COVID-19의 확산으로 인해 공공 장소에서의 마스크 착용 여부의 중요성이 더해짐에 따라 사람 얼굴의 이미지만으로 마스크 착용 여부, 나이, 성별 분류가 가능한 모델을 구현하고자 한다
## 활용 장비 및 재료(개발 환경, 협업 tool 등)
- vscode, jupyter notebook
- v100 서버
- 구글공유드라이브, github
## 프로젝트 구조 및 사용 데이터셋의 구조도(연관도)
![enter image description here](https://github.com/boostcampaitech3/level1-image-classification-level1-recsys-07/blob/main/assets/2.png)
## 기대 효과
- **기술가치**
	- 얼굴인식과 분류 task의 성능을 높임으로써 face-landmark extraction, face recognition, face-expression-recognition등 얼굴 인식과 관련된 분류 task에 재활용될 수 있다.
	- CNN기반 모델과 Transformer 기반의 모델을 활용함으로써 다양한 도메인의 이미지 처리에 이용해볼 수 있다.
- **시장가치**
	- 적은 인적자원으로도 COVID-19와 같은 전염병의 전파를 막기 위한 마스크 유무 판별 가능
	은행과 같은 얼굴인식이 중요한 기관에 높은 성능의 알고리즘을 제공할 수 있다.
# 팀 구성 및 역할
- 박동재 (팀장) : HandlingImbalance Problem / Modeling / HyperParameter tuning
- 박병윤 : HandlingImbalance Problem/ Modeling /HyperParameter tuning
- 김문정 :  EDA/ Modeling / data purification
- 이동훈 : EDA / Data curation /  Data purification
- 용자윤 : EDA / Modeling /  HyperParameter tuning
# 수행 절차 및 방법
![enter image description here](https://github.com/boostcampaitech3/level1-image-classification-level1-recsys-07/blob/main/assets/3.png)
# 수행 결과
- 탐색적 분석 및 전처리 (학습데이터 소개)
    - 데이터의 분포 조사
![enter image description here](https://github.com/boostcampaitech3/level1-image-classification-level1-recsys-07/blob/main/assets/image.png)
         - 60대 이상의 데이터가 매우 적은 것을 확인할 수 있음
         - 특히 60대 이상의 데이터가 60세만 있음
- 모델 개요
	- Vision Tranformer (ViT) 기반
		- NLP Task에서 주로 사용되었던 Transformer를 Image patch들의 sequence에 적용한 모델
		- Patch Embedding과 Transformer를 통해 Class 정보와 나뉜 이미지 영역에 Attention이 가능하며 Self-Attention 메커니즘을 통해 Global 한 영역의 처리가 가능하다
		- 사용 알고리즘: vit_tiny_patch16_224, vit_small_patch32_224, vit_base_patch16_224, vit_large_patch16_224, deit_base_patch16_224, 
				swin_base_patch4_window7_224, vit_small_r26_s32_224_in21k, vit_base_r50_s16_224_in21k
		- 성능개선 방법 : cutmix, mixup, k fold ensemble, batch accumulation, early stopping, test time augmentation
	- EfficientNet 기반
		- 모델의 깊이, 너비, 입력 이미지의 크기를 효율적으로 조절하는 compound scaling 방법을 제안한 모델
		- 사용 알고리즘 : resnet18, resnet34, resnet26, resnet50, resnet18d, resnet34d, resnet26d, skresnet18
		- 성능개선 방법 :mixup, batch accumulation, early stopping
	- Resnet 기반
 		- VGGnet을 기반으로 한 모델, 층의 깊이가 깊어질수록 발생하는 기울기 소실/ 폭발문제를 skip connection을 이용한 residual learning을 통해 해결
		- 사용 알고리즘 : efficientnet_b0, efficientnet_b1, efficientnet_b3, efficientnet_b5, 
				efficientnet_b7, efficientnetv2_rw_t, efficientnetv2_rw_s, efficientnetv2_rw_m
		- 성능개선 방법 : cutmix, mixup, batch accumulation, early stopping
- 모델 선정 및 분석
    - timm 모듈을 활용하여, 데이터를 고정시키고, 해당하는 데이터에 가장 잘 맞는 모델  선택
![enter image description here](https://github.com/boostcampaitech3/level1-image-classification-level1-recsys-07/blob/main/assets/7.png)
    - 해당하는 데이터에 가장 좋은 성능을 보인 3가지의 모델을 기반으로 적절한 데이터 처리 및 하이퍼 파라미터 튜닝
    - 모델 : vit_small_r26_s32_224_in21k
        - Optimizer : SGD
        - batch size : 64
        - Learning rate : 0.001
        - Loss function : Label Smoothing Loss
        - Data Augmentation : 0.35 Mixup / 0.35 Cutmix / 0.3 Original Data
    - 모델: efficientnetv2_rw_t
        - Optimizer : Adam
        - batch size : 128
        - Learning rate : 0.00025
        - Loss function : Weighted Cross Entropy Loss
        - Augmentation : 0.5 Mixup / 0.5 Original Data
    - 모델: resnet34
        - Optimizer : Adam
        - batch size : 128
        - Learning rate : 0.0001
        - Loss function : Focal Loss
        - Data Augmentation : 0.25 Mixup / 0.25 Cutmix / 0.5 Original Data
    - 그 외 image size, weight decay, learning rate decay step, early stopping 등의 hyperparameter는 고정하여 실험 진행
- 3가지의 모델 각각을 최고의 성능 조합으로 앙상블 이용
	- 성능 best 3 모델
		- Valid dataset 기준
		- 리더보드 기준
- 모델 평가 및 개선
	- 모델 평가기준 : 자체적으로 나눈 valid(사람기준으로 split) 데이터에 대한 f1-score를 기준으로 평가
		- 개선방향 : 팀 노션을 이용해 각각의 config-file을 공유 간단한 back-bone모델과 중요 파라미터를 명시하고 성능을 그래프로 확인 후 개선점 논의
- 시연 결과
    - 모델 성능
![enter image description here](https://github.com/boostcampaitech3/level1-image-classification-level1-recsys-07/blob/main/assets/8.png)
f1 socre : 0.73, accuracy : 79.33으로 private leaderboard에서 19위 달성
결과물 사진/ 시연 동영상 등 선택적 포함
# References (코드 및 자료)
https://github.com/rwightman/pytorch-image-models
