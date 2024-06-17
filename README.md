# AI CUPID Project 💘

소개팅을 위한 닮은꼴 매칭 실험을 진행했습니다.

## 프로젝트 소개

AI CUPID 프로젝트는 인공지능 얼굴인식 기술을 활용하여 소개팅을 위한 닮은꼴 매칭 실험을 진행하는 것을 목표로 합니다. 인공지능 얼굴인식 기술은 얼굴의 기하학적 구조와 표정을 분석하여 얼굴을 다른 물체와 구분합니다. 이 과정에서 얼굴 랜드마크를 식별하는 것이 핵심입니다. 랜드마크 포인트에는 두 눈 사이의 거리, 이마에서 턱까지의 거리, 코와 입 사이의 거리, 안와의 깊이, 광대뼈의 모양, 입술, 귀, 턱의 윤곽 등이 포함됩니다.

## 실험 과정

### 얼굴 인식 및 매칭

1. **얼굴 랜드마크 식별 및 매칭**:
    - 랜드마크 포인트를 기준으로 얼굴 영역을 회전시키고, 얼굴 매칭이 가능한 상태로 변경합니다.
    - 이후 얼굴 영역을 임베딩 과정을 거쳐 N차원의 특징 벡터로 표현합니다.
    - 입력된 이미지 속 얼굴이 누구인지 판단하기 위해 특징 벡터 간의 유사도를 계산합니다.

2. **데이터 전처리**:
    - 총 54명의 참여자를 모집하여, 참여자의 얼굴 사진을 유사도 측정에 용이하도록 사진 크롭 및 배경 제거 등의 데이터 전처리를 진행했습니다.

### 허깅페이스(Hugging Face) 실험

첫 번째 실험은 허깅페이스에서 진행되었습니다. 하지만 허깅페이스에서 전체 54명의 얼굴을 분석하는 코드를 돌리기에는 RunTime이 너무 오래 걸리는 문제가 발생했습니다. 그래서 허깅페이스는 1:1 매칭에 사용하고, Colab에서 전체 결과를 계산하기로 했습니다.

- **Huggingface Demo**:
  [Huggingface Demo 링크](https://huggingface.co/spaces/suinY00N/CupidAI)

허깅페이스에서는 웹캠을 이용하여 실시간으로 유사도를 분석할 수 있습니다. Submit 버튼을 누르면 DeepFace 모델을 통해 얼굴 유사도를 측정하고, 두 사람의 얼굴 이미지를 각각 업로드합니다. 유사도를 거리 기반으로 계산하기 때문에 거리가 가까울수록 유사한 것으로 판단하며, 설정한 임계치에 따라 다른 값이 출력되도록 하였습니다.

### Google Colab 실험

두 번째 실험은 Google Colab을 이용했습니다. Colab에서 이미지 파일을 기반으로 얼굴 인코딩을 진행하고, 인코딩을 통해 모든 남자와 여자 간의 거리를 계산한 후, 거리 순으로 매칭을 수행했습니다.

- **Colab Demo**:
  [Colab Demo 링크](https://colab.research.google.com/drive/1048h_3ziEUErCaq3Sdn0TzB2OGeo9Xr_?usp=sharing)

# Code
## 얼굴 인식을 위한 라이브러리 설치 및 유사도 계산

### 라이브러리 설치


```python
!pip install face_recognition
!pip install opencv-python
!pip install pandas
!pip install deepface

import face_recognition
import cv2
import numpy as np
import os
import dlib
import pandas as pd
from deepface import DeepFace

# CUDA를 사용하지 않도록 설정
dlib.DLIB_USE_CUDA = True

import os
import face_recognition

# 이미지 폴더 경로
image_folder = '/content/drive/MyDrive/images_3'

# 남자와 여자 이미지 파일명 리스트 생성
male_files = [f'{i}_M.jpeg' for i in range(1, 33)]
female_files = [f'{i}_F.jpeg' for i in range(1, 23)]

# 이미지 파일 경로 확인
all_files = male_files + female_files
missing_files = [file for file in all_files if not os.path.exists(os.path.join(image_folder, file))]

if missing_files:
    print("다음 파일들이 존재하지 않습니다:")
    for file in missing_files:
        print(file)
else:
    print("모든 파일이 존재합니다.")

# 얼굴 인코딩을 저장할 딕셔너리
male_encodings = {}
female_encodings = {}

# 남자 얼굴 인코딩
for file in male_files:
    img_path = os.path.join(image_folder, file)
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)
    if encoding:
        male_encodings[file] = encoding[0]

# 여자 얼굴 인코딩
for file in female_files:
    img_path = os.path.join(image_folder, file)
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)
    if encoding:
        female_encodings[file] = encoding[0]

# 모든 남자와 여자 간의 거리 계산
distances = []
for male, male_encoding in male_encodings.items():
    for female, female_encoding in female_encodings.items():
        distance = face_recognition.face_distance([male_encoding], female_encoding)[0]
        distances.append((male, female, distance))

# 거리 기준으로 정렬 (거리가 작은 순서대로)
distances.sort(key=lambda x: x[2])

# 매칭 결과와 이미 매칭된 남녀를 추적할 집합
matches = []
matched_males = set()
matched_females = set()

# 거리 순서대로 매칭 수행
for male, female, distance in distances:
    if male not in matched_males and female not in matched_females:
        matches.append((male, female, distance))
        matched_males.add(male)
        matched_females.add(female)
    # 모든 남성과 여성이 매칭되면 종료
    if len(matched_males) == len(male_encodings) or len(matched_females) == len(female_encodings):
        break

# 매칭 결과 출력
for male, female, distance in matches:
    print(f"남자 {male}와 여자 {female} 매칭 (거리: {distance})")

from deepface import DeepFace

# 모든 남자와 여자 간의 거리 계산
distances = []
for male, male_encoding in male_encodings.items():
    for female, female_encoding in female_encodings.items():
        # DeepFace를 사용한 유사도 계산
        result = DeepFace.verify(img1_path=os.path.join(image_folder, male), img2_path=os.path.join(image_folder, female))
        distance = result['distance']
        distances.append((male, female, distance))

# 거리 기준으로 정렬 (거리가 작은 순서대로)
distances.sort(key=lambda x: x[2], reverse=True)

# 매칭 결과와 이미 매칭된 남녀를 추적할 집합
matches = []
matched_males = set()
matched_females = set()

# 거리 순서대로 매칭 수행
for male, female, distance in distances:
    if male not in matched_males and female not in matched_females:
        matches.append((male, female, distance))
        matched_males.add(male)
        matched_females.add(female)
    # 모든 남성과 여성이 매칭되면 종료
    if len(matched_males) == len(male_encodings) or len(matched_females) == len(female_encodings):
        break

# 매칭 결과 출력
for male, female, distance in matches:
    print(f"남자 {male}와 여자 {female} 매칭 (거리: {distance})")


1차 실험 결과에서는 모든 남성과의 매칭에서 외국인인 여자 8번과 9번이 가장 유사하지 않은 사람으로 선정되었습니다. 인공지능의 얼굴 인식 로직에 따르면, 인종이 다를 경우 유사도가 낮을 수밖에 없습니다. 또한, 가장 유사도가 높은 사람으로는 1번이 무려 11번 중복되어 매칭되었습니다.

### 2차 매칭 실험

원활한 데이트 매칭을 위해, 매칭이 겹치지 않도록 코드를 수정하여 2차 매칭을 진행했습니다. 다음은 코드를 수정한 2차 매칭 결과입니다.

## 인간 매칭 실험

마지막으로, 인공지능 매칭 결과와 비교하기 위해 인간의 매칭을 시도했습니다. 이를 위해 팀원 3명이 사진을 기반으로 첫 인상을 보고 닮은 사람 매칭을 취합하였습니다.

---
# 
### 기여자

- 팀장 윤수인
- 팀원 한예슬
- 팀원 김찬기

### 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.

### 문의

프로젝트에 대한 문의사항이 있으시면 [dbsdev98@gmail.com]으로 연락해 주세요.

