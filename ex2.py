# OpenCv & torch
import cv2
import torch

# 시간 라이브러리
import datetime

# YOLOv5 모델 불러오기
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l',
                            device='cuda:0' if torch.cuda.is_available() else 'cpu')  # 예측 모델
yolo_model.classes = [0]  # 예측 클래스 (index의 값 0번이 사람)

# 웹캠 비디오 캡처 설정 - 기본 웹캡
cap = cv2.VideoCapture(0)

# 동영상 파일 저장 설정
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (640, 480))  # width와 height는 적절한 값으로 조정

while True:
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        break

    current_time = datetime.datetime.now().strftime("%H시%M분%S초")  # 현재 시간 계산
    current_second = cap.get(cv2.CAP_PROP_POS_MSEC) / 100

    # 동영상 시간을 분과 초로 변환
    video_minutes, video_seconds = divmod(int(current_second), 60)

    # YOLOv5 모델로 객체 감지 수행
    results = yolo_model(frame)
    results_refine = results.pandas().xyxy[0].values
    nms_human = len(results_refine)

    # 결과 출력 - *초 : *명
    print(f"현재시간 : {current_time} - {video_minutes}분 {video_seconds}초 - 감지된 사람 수 : {nms_human}명")
    if nms_human > 0:
        for bbox in results_refine:
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))

            frame = cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 3)

    # 화면에 영상 출력
    cv2.imshow("Real-time Object Detection", frame)

    # 동영상 파일에 프레임 저장
    out.write(frame)

    # 키 입력 대기
    key = cv2.waitKey(1)

    # 'q' 키를 누르면 종료
    if key == ord("q") or key == 27:  # 27은 ESC 키의 ASCII 코드
        break

# 종료 시 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()