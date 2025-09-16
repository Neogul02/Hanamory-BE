# YOLOv5 Flask API on Azure

🌸 **꽃다발 인식 YOLOv5 모델을 Flask API로 서빙하고 Azure에 배포하는 프로젝트**

## 📋 프로젝트 개요

이 프로젝트는 YOLOv5로 학습된 꽃다발 인식 모델을 Flask API로 서빙하고, Docker 컨테이너로 패키징하여 Azure Container Instances에 배포합니다.

## 🏗️ 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   Flask API     │───▶│   YOLOv5 Model  │
│                 │    │  (Docker)       │    │   (best.pt)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Azure Container │
                       │   Instances     │
                       └─────────────────┘
```

## 📁 프로젝트 구조

```
yolo-flask-azure/
├── backend/
│   ├── app.py              # Flask API 서버
│   └── models/
│       └── best.pt         # 학습된 YOLOv5 모델
├── Dockerfile              # Docker 이미지 빌드 설정
├── .gitignore             # Git 제외 파일 설정
└── README.md              # 프로젝트 문서
```

## 🚀 빠른 시작

### 1. 로컬 실행

```bash
# 가상환경 생성 및 활성화
python -m venv yolo_env
# Windows
yolo_env\Scripts\activate
# macOS/Linux
source yolo_env/bin/activate

# 필요한 라이브러리 설치
pip install flask flask-cors torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics pillow opencv-python-headless numpy pyyaml requests psutil

# Flask 서버 실행
cd backend
python app.py
```

### 2. Docker 실행

```bash
# Docker 이미지 빌드
docker build -t yolo-flask-app .

# Docker 컨테이너 실행
docker run -p 5000:5000 yolo-flask-app
```

## 📡 API 엔드포인트

### `GET /`
기본 상태 확인

### `GET /health`
서버 헬스체크 (CPU, 메모리 사용률 포함)

### `POST /predict`
이미지 업로드하여 객체 탐지 결과 이미지 반환

**요청:**
```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/predict
```

**응답:** 탐지 결과가 표시된 이미지 파일

### `POST /predict-json`
이미지 업로드하여 JSON 형태로 탐지 결과 반환

**응답:**
```json
{
  "predictions": [
    {
      "class": "flower_bouquet",
      "confidence": 0.85,
      "bbox": [100, 150, 300, 400]
    }
  ],
  "count": 1,
  "image_size": [416, 416],
  "model_name": "flower2_yolov5"
}
```

## 🐳 Docker 특징

- **Base Image:** Python 3.8-slim (NumPy 호환성 최적화)
- **CPU 최적화:** PyTorch CPU 버전 사용
- **경량화:** 필요한 패키지만 설치
- **Health Check:** 자동 서비스 상태 모니터링

## ☁️ Azure 배포

### 필요한 도구
- Azure CLI
- Docker

### 배포 과정
1. Azure Container Registry 생성
2. Docker 이미지 빌드 및 푸시
3. Azure Container Instances에 배포

```bash
# Azure 로그인
az login

# 리소스 그룹 생성
az group create --name yolo-flask-rg --location koreacentral

# Container Registry 생성
az acr create --resource-group yolo-flask-rg --name yoloflaskacr --sku Basic

# Docker 이미지 태그 및 푸시
docker tag yolo-flask-app yoloflaskacr.azurecr.io/yolo-flask-app:latest
docker push yoloflaskacr.azurecr.io/yolo-flask-app:latest

# Container Instance 생성
az container create \
    --resource-group yolo-flask-rg \
    --name yolo-flask-container \
    --image yoloflaskacr.azurecr.io/yolo-flask-app:latest \
    --cpu 1 --memory 2 \
    --ports 5000
```

## 🔧 성능 최적화

### CPU 사용률 최적화
- OpenMP, MKL, PyTorch 스레드 수 제한 (1개)
- 이미지 크기 자동 최적화 (최대 800px)
- 메모리 관리 (가비지 컬렉션)
- 요청 큐잉 시스템

### 설정값
- **이미지 크기:** 416x416
- **Confidence 임계값:** 0.5
- **최대 검출 수:** 50개
- **CPU 임계값:** 85% (초과 시 요청 거부)

## 🐛 문제 해결

### 일반적인 문제

1. **NumPy 호환성 오류**
   - Python 3.8 사용으로 해결

2. **결과 이미지를 찾을 수 없음**
   - 모든 `results*` 폴더에서 자동 검색하도록 구현

3. **CPU 사용률 99% 문제**
   - 스레드 제한 및 메모리 관리로 해결

## 📈 모니터링

Flask API는 다음 메트릭을 제공합니다:
- CPU 사용률
- 메모리 사용률  
- 모델 로딩 상태
- 큐 크기

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 감사의 말

- [YOLOv5](https://github.com/ultralytics/yolov5) - Ultralytics
- [Flask](https://flask.palletsprojects.com/) - Pallets
- [PyTorch](https://pytorch.org/) - Facebook AI Research
