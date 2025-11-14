# [Hanamory] 꽃다발 인식 AI 서비스

<div align="center">
  
**YOLOv5를 활용한 꽃다발 이미지 인식 및 객체 탐지 REST API 서비스**

</div>

## 👋 배포 주소

> **Hanamory API** : [API 서버](https://port-0-hanamory-be-m3e7qqcm7e3f3df2.sel4.cloudtype.app/) <br> > **프론트엔드** : [Hanamory FE](https://github.com/Neogul02/Hanamory-FE) <br> > **백엔드** : [Hanamory BE](https://github.com/Neogul02/Hanamory-BE)

## � 팀 소개

|                                                         최진형                                                         |
| :--------------------------------------------------------------------------------------------------------------------: |
| <img width="160" alt="최진형" src="https://github.com/user-attachments/assets/9e0ee844-d700-47b3-8c27-ec861e4bf11a" /> |
|                                     [@Choe JinHyeong](https://github.com/Neogul02)                                     |
|                                                  Fullstack Developer                                                   |

## � 프로젝트 소개

Hanamory는 YOLOv5 커스텀 모델을 활용하여 꽃다발 이미지를 인식하고 분석하는 AI 서비스예요. 사용자가 업로드한 이미지에서 꽃다발을 자동으로 탐지하고, 바운딩 박스가 표시된 이미지와 JSON 형태의 상세 분석 결과를 제공해요. Docker 컨테이너화를 통해 Cloudtype에 배포되어 안정적인 서비스를 제공하고 있어요.

<br>

## ✨ 서비스 핵심 기능

### 1. 이미지 기반 꽃다발 객체 탐지

업로드된 이미지에서 YOLOv5 모델이 꽃다발을 자동으로 인식하고 바운딩 박스로 표시해요.

### 2. 듀얼 API 엔드포인트 제공

- **`/predict`**: 바운딩 박스가 그려진 결과 이미지를 반환
- **`/predict-json`**: 클래스명, 신뢰도, 좌표 정보를 JSON으로 반환

### 3. 이미지 최적화 및 서버 안정성

- 이미지 자동 리사이징 (최대 800px) 및 품질 최적화 (JPEG 85%)
- CPU 사용률 모니터링을 통한 과부하 방지
- 파일 크기 제한 (최대 10MB) 및 자동 임시 파일 정리

### 4. 헬스체크 및 모니터링

실시간 CPU/메모리 사용률과 모델 상태를 확인할 수 있는 `/health` 엔드포인트를 제공해요.

---

## 💻 Tech Stack

| 구분                 | 기술                                                                                                                                                                                                                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **AI/ML**            | ![YOLOv5](https://img.shields.io/badge/YOLOv5-00FFFF?style=flat-square&logo=yolo&logoColor=black) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![Ultralytics](https://img.shields.io/badge/Ultralytics-000000?style=flat-square) |
| **Backend**          | ![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white) ![Gunicorn](https://img.shields.io/badge/Gunicorn-499848?style=flat-square&logo=gunicorn&logoColor=white)                                                                                 |
| **Image Process**    | ![Pillow](https://img.shields.io/badge/Pillow-3776AB?style=flat-square) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)                                                                                                                |
| **Containerization** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) ![Docker Compose](https://img.shields.io/badge/Docker_Compose-2496ED?style=flat-square&logo=docker&logoColor=white)                                                                    |
| **Deployment**       | ![Cloudtype](https://img.shields.io/badge/Cloudtype-000000?style=flat-square) ![AWS EC2](https://img.shields.io/badge/AWS_EC2-FF9900?style=flat-square&logo=amazon-aws&logoColor=white)                                                                                                    |
| **Monitoring**       | ![psutil](https://img.shields.io/badge/psutil-3776AB?style=flat-square)                                                                                                                                                                                                                    |

## 🚀 설치 및 실행

### 로컬 환경에서 실행

```bash
# 저장소 클론
git clone https://github.com/Neogul02/Hanamory-BE.git
cd Hanamory-BE

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
pip install ultralytics pillow opencv-python-headless numpy pyyaml requests psutil

# Flask 서버 실행
cd backend
python app.py
```

### Docker 실행

```bash
# Docker 이미지 빌드
docker build -t hanamory-api .

# Docker 컨테이너 실행
docker run -p 5000:5000 hanamory-api

# Docker Compose 실행
docker-compose up -d
```

---

## 📡 API 엔드포인트

### `GET /`

기본 상태 확인 및 서비스 정보 조회

### `GET /health`

서버 헬스체크 (CPU, 메모리 사용률, 모델 로딩 상태 포함)

### `POST /predict`

이미지 업로드 후 바운딩 박스가 표시된 결과 이미지 반환

**요청 예시:**

```bash
curl -X POST -F "image=@flower.jpg" https://port-0-hanamory-be-m3e7qqcm7e3f3df2.sel4.cloudtype.app/predict
```

**응답:** 객체 탐지 결과가 표시된 이미지 파일

### `POST /predict-json`

이미지 업로드 후 JSON 형태로 탐지 결과 반환

**응답 예시:**

```json
{
  "predictions": [
    {
      "class": "flower_bouquet",
      "confidence": 0.8542,
      "bbox": [125.45, 180.23, 340.67, 455.89]
    }
  ],
  "count": 1,
  "image_size": [640, 640],
  "model_name": "flower2_yolov5"
}
```

---

## � 최적화 및 특징

### 이미지 처리 최적화

- 최대 이미지 크기 제한 (800px)
- JPEG 품질 85% 자동 압축
- RGB 색상 모드 자동 변환
- 업로드 파일 크기 제한 (10MB)

### CPU 리소스 관리

- CPU 사용률 모니터링 (95% 초과 시 요청 거부)
- PyTorch/OpenMP/MKL 스레드 수 제한 (1개)
- 자동 메모리 가비지 컬렉션
- 임시 파일 자동 정리

### 모델 설정

- **이미지 크기:** 640x640
- **신뢰도 임계값:** 0.5
- **최대 검출 수:** 50개
- **디바이스:** CPU 전용

---

## ☁️ Cloudtype 배포

이 프로젝트는 Cloudtype 플랫폼에 배포되어 운영되고 있어요.

### 배포 환경

- **플랫폼:** Cloudtype
- **컨테이너:** Docker
- **리소스:** CPU 1 Core, Memory 2GB
- **포트:** 5000

### 환경 변수

```bash
FLASK_ENV=production
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
TORCH_NUM_THREADS=1
TORCH_HOME=/tmp/torch_cache
```

---

## 📂 프로젝트 구조

```
Hanamory-BE/
├── backend/
│   ├── app.py              # Flask API 서버
│   ├── models/
│   │   └── best.pt         # YOLOv5 커스텀 학습 모델
│   └── static/
│       └── uploads/        # 임시 업로드 디렉토리
├── Dockerfile              # Docker 이미지 설정
├── docker-compose.yml      # Docker Compose 설정
├── requirements.txt        # Python 의존성
└── README.md
```

---

## � 개발 과정

### YOLOv5 모델 커스텀 학습

꽃다발 데이터셋으로 YOLOv5 모델을 파인튜닝하여 `best.pt` 모델 생성

### Flask API 구축

`torch.hub.load()`를 통한 모델 로딩 및 전역 변수 관리로 메모리 효율성 최적화

### Docker 컨테이너화

Python 3.8-slim 베이스로 경량화된 이미지 구성 및 헬스체크 기능 구현

### 클라우드 배포

Cloudtype 환경에 맞춘 캐시 디렉토리 설정 및 리소스 제한 최적화

---

## 📈 모니터링

`/health` 엔드포인트를 통해 다음 메트릭을 실시간으로 확인할 수 있어요:

- CPU 사용률
- 메모리 사용률
- 모델 로딩 상태
- 모델 파일 존재 여부

---

## 🙏 Reference

- [YOLOv5](https://github.com/ultralytics/yolov5) - Ultralytics
- [Flask](https://flask.palletsprojects.com/) - Pallets
- [PyTorch](https://pytorch.org/) - Facebook AI Research
