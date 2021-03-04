# dio-ai
포자랩스 딥러닝 샘플 자동 생성 프로젝트

## 개발환경 구성
### 프로젝트 개발 환경 구성
- 프로젝트별로 별도 가상환경 (virtualenv)를 구성해주세요.
- PR로 디펜던시가 변경될 경우 `poetry update`로 디펜던시를 업데이트해주세요.

1. Python 환경 구성
    - pyenv 설치 및 python 환경 구성 (Python 버전: 3.7)
    - 가상 환경 구성 (`python3 -m venv ...`)
2. 프로젝트 환경 구성
    - poetry 설치 (`pip3 install poetry`)
    - 디펜던시 설치 (`poetry install`)
    - pre-commit 구성 (`pre-commit install`)
    
### GPU 인식 테스트
```python
# Tensorflow
import tensorflow as tf
# GPU 인식 확인
print(tf.test.is_gpu_available())
# 인식되는 GPU 출력
print(tf.config.list_physical_devices())

# PyTorch
import torch
print(torch.cuda.is_available())
# 디바이스 개수 확인
print(torch.cuda.device_count())
# 디바이스 이름 확인 (e.g. GeForce RTX 2080 Ti)
print(torch.cuda.get_device_name())
```

## 기타
- PyTorch 설치 문서: [START LOCALLY](https://pytorch.org/get-started/locally/#start-locally)
- poetry 멀티 플랫폼 설치 문서: [Multiple constraints dependencies](https://python-poetry.org/docs/dependency-specification/#using-environment-markers)
