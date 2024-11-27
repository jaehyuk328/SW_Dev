import cv2
import os
from ultralytics import YOLO

# YOLO 모델 로드 (yolo11n.pt 모델 경로)
model = YOLO("yolo11n.pt")

folder_path = r"C:\Users\cic\Desktop\Local_Repo\Image_dataset"
file_names = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

# 객체 탐지 함수 정의
def detect_objects(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    
    # 모델을 통해 객체 탐지 결과 얻기
    results = model(image_path)
    
    # 탐지된 객체들에 대해 바운딩 박스와 라벨 그리기
    for result in results:  # 탐지된 객체마다 반복
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
            confidence = box.conf[0]  # 신뢰도 점수
            label = box.cls[0]  # 클래스 라벨 인덱스
            label_text = f"{model.names[int(label)]} {confidence:.2f}"
            
            # 바운딩 박스 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 라벨 그리기
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 결과 이미지 보여주기
    cv2.imshow('YOLO Detections', image)

# 이미지 목록 순차적으로 탐지
for image_name in file_names:
    image_path = os.path.join(folder_path, image_name)
    detect_objects(image_path)
    
    # 'q' 키를 누르면 다음 이미지로 넘어감
    key = cv2.waitKey(0)  # 키 입력 대기
    if key == ord('q'):
        continue  # 'q' 키가 눌리면 다음 이미지로 넘어감
    elif key == 27:  # ESC 키를 누르면 종료
        break

cv2.destroyAllWindows()