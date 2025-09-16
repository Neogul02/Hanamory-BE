import os
from pathlib import Path
import uuid
import gc
import psutil
import time
import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

# CPU 최적화 설정
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TORCH_NUM_THREADS'] = '1'
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'best.pt'
UPLOAD_DIR = BASE_DIR / 'static' / 'uploads'
RESULT_DIR = BASE_DIR / 'static' / 'results'

# 설정값
IMGSZ = (640, 640)
CONF_THRES = 0.5
MAX_DET = 50
MAX_IMAGE_SIZE = 800

# 디렉토리 생성
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)

# 전역 모델 변수
global_model = None

def load_model():
    """모델 로딩"""
    global global_model
    if global_model is None:
        try:
            torch.set_num_threads(1)
            print(f"모델 로딩 시도: {MODEL_PATH}")

            if MODEL_PATH.exists():
                # 학습된 모델 로딩
                global_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                            path=str(MODEL_PATH),
                                            force_reload=False,
                                            trust_repo=True)
                print("✓ 학습된 flower2 모델 로딩 완료")
            else:
                # 기본 모델 사용
                global_model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                                            pretrained=True,
                                            trust_repo=True)
                print("✓ 기본 YOLOv5s 모델 로딩 완료")

            global_model.to('cpu')
            global_model.eval()
            global_model.conf = CONF_THRES
            global_model.max_det = MAX_DET

        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            global_model = None
    return global_model

def optimize_image(image_path, max_size=MAX_IMAGE_SIZE):
    """이미지 최적화"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            img.save(image_path, 'JPEG', quality=85, optimize=True)
            return True
    except Exception as e:
        print(f"이미지 최적화 실패: {e}")
        return False

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "YOLOv5 Flower Detection API",
        "version": "Final",
        "model": "flower2_yolov5" if MODEL_PATH.exists() else "yolov5s",
        "status": "running"
    })

@app.route('/health', methods=['GET'])
def health():
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_percent = psutil.virtual_memory().percent

    return jsonify({
        "status": "healthy",
        "cpu_usage": f"{cpu_percent:.1f}%",
        "memory_usage": f"{memory_percent:.1f}%",
        "model_loaded": global_model is not None,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists()
    })

@app.route('/predict', methods=['POST'])
def predict():
    # CPU 사용률 체크
    cpu_percent = psutil.cpu_percent(interval=0.1)
    if cpu_percent > 95.0:
        return jsonify({'error': 'CPU 사용률이 높습니다. 잠시 후 다시 시도해주세요.'}), 503

    if 'image' not in request.files:
        return jsonify({'error': '이미지가 필요합니다.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400

    # 파일명 생성
    filename = f"{uuid.uuid4().hex}.jpg"
    input_path = UPLOAD_DIR / filename
    result_path = RESULT_DIR / filename

    try:
        # 파일 크기 체크
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)

        if file_size > 10 * 1024 * 1024:  # 10MB
            return jsonify({'error': '파일 크기가 너무 큽니다. (최대 10MB)'}), 413

        # 파일 저장 및 최적화
        file.save(str(input_path))
        if not optimize_image(input_path):
            return jsonify({'error': '이미지 최적화 실패'}), 400

        # 모델 로딩
        model = load_model()
        if model is None:
            return jsonify({'error': '모델 로딩 실패'}), 500

        # 예측 실행
        with torch.no_grad():
            results = model(str(input_path), size=IMGSZ[0])

        # 결과 이미지 저장 (직접 저장)
        results.save(save_dir=str(RESULT_DIR))

        # 결과 파일 찾기 - 모든 results 폴더에서 검색
        result_path = None

        # results, results2, results3 등 모든 폴더에서 검색
        results_dirs = []
        base_static_dir = BASE_DIR / 'static'

        # 모든 results 관련 폴더 찾기
        for item in base_static_dir.iterdir():
            if item.is_dir() and item.name.startswith('results'):
                results_dirs.append(item)

        # 가장 최근에 생성된 이미지 파일 찾기
        all_result_files = []
        for results_dir in results_dirs:
            result_files = list(results_dir.glob("*.jpg"))
            all_result_files.extend(result_files)

        if all_result_files:
            # 가장 최근 생성된 파일 선택
            latest_file = max(all_result_files, key=lambda x: x.stat().st_mtime)

            # 예측 가능한 이름으로 복사
            final_result_path = RESULT_DIR / filename

            # 결과 디렉토리가 없으면 생성
            RESULT_DIR.mkdir(parents=True, exist_ok=True)

            # 파일 복사
            import shutil
            shutil.copy2(str(latest_file), str(final_result_path))
            result_path = final_result_path

            print(f"결과 이미지 복사: {latest_file} -> {final_result_path}")

        if result_path and result_path.exists():
            return send_file(result_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': '예측 결과를 찾을 수 없습니다.'}), 500

    except Exception as e:
        print(f"예측 오류: {e}")
        return jsonify({'error': f'예측 중 오류 발생: {str(e)}'}), 500
    finally:
        # 임시 파일 정리
        if input_path.exists():
            try:
                input_path.unlink()
            except:
                pass
        gc.collect()

@app.route('/predict-json', methods=['POST'])
def predict_json():
    # CPU 사용률 체크
    cpu_percent = psutil.cpu_percent(interval=0.1)
    if cpu_percent > 85.0:
        return jsonify({'error': 'CPU 사용률이 높습니다. 잠시 후 다시 시도해주세요.'}), 503

    if 'image' not in request.files:
        return jsonify({'error': '이미지가 필요합니다.'}), 400

    file = request.files['image']
    filename = f"{uuid.uuid4().hex}.jpg"
    input_path = UPLOAD_DIR / filename

    try:
        file.save(str(input_path))
        if not optimize_image(input_path):
            return jsonify({'error': '이미지 최적화 실패'}), 400

        model = load_model()
        if model is None:
            return jsonify({'error': '모델 로딩 실패'}), 500

        # 예측 실행
        with torch.no_grad():
            results = model(str(input_path), size=IMGSZ[0])

        predictions = []

        if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
            detections = results.xyxy[0].cpu().numpy()

            for detection in detections:
                if len(detection) >= 6:
                    x1, y1, x2, y2, conf, cls = detection[:6]

                    if conf >= CONF_THRES:
                        predictions.append({
                            'class': model.names[int(cls)] if hasattr(model, 'names') else f'class_{int(cls)}',
                            'confidence': round(float(conf), 4),
                            'bbox': [round(float(x1), 2), round(float(y1), 2),
                                   round(float(x2), 2), round(float(y2), 2)]
                        })

        return jsonify({
            'predictions': predictions,
            'count': len(predictions),
            'image_size': IMGSZ,
            'model_name': 'flower2_yolov5' if MODEL_PATH.exists() else 'yolov5s'
        })

    except Exception as e:
        print(f"predict-json 오류: {e}")
        return jsonify({'error': f'예측 중 오류 발생: {str(e)}'}), 500
    finally:
        if input_path.exists():
            try:
                input_path.unlink()
            except:
                pass
        gc.collect()

if __name__ == '__main__':
    print("🚀 YOLOv5 Flask Server 시작...")
    print(f"📊 설정: 이미지 크기={IMGSZ}, 신뢰도={CONF_THRES}")
    print(f"🎯 모델: {'학습된 flower2 모델' if MODEL_PATH.exists() else '기본 YOLOv5s 모델'}")

    # 모델 미리 로딩
    load_model()

    app.run(host='0.0.0.0', port=5000, debug=False)
