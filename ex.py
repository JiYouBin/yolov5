# OpenCv & torch
import cv2
import torch
import datetime
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware
import json
import asyncio
import login


app = FastAPI()

# YOLOv5 모델 불러오기
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l',
                            device='cuda:0' if torch.cuda.is_available() else 'cpu')  # 예측 모델
yolo_model.classes = [0]  # 예측 클래스 (index의 값 0번이 사람)

# 웹캠 비디오 캡처 설정 - 기본 웹캠
cap = cv2.VideoCapture(0)

templates = Jinja2Templates(directory="templates")


@app.get("/video_feed")
async def video_feed():
    async def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLOv5 모델로 객체 감지 수행
            results = yolo_model(frame)
            results_refine = results.pandas().xyxy[0].values
            nms_human = len(results_refine)

            # 화면에 객체 감지 정보를 출력
            if nms_human > 0:
                for bbox in results_refine:
                    start_point = (int(bbox[0]), int(bbox[1]))
                    end_point = (int(bbox[2]), int(bbox[3]))
                    frame = cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 3)

            __, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n' )
            
            await asyncio.sleep(0.1)
            
            
            
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/Info", response_class=HTMLResponse)
async def get_video(request: Request):
    return templates.TemplateResponse('ex.html', {"request": request})

import json

@app.get("/api/json_data")
async def get_json_data():
    async def json_data_generator():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLOv5 모델로 객체 감지 수행
            results = yolo_model(frame)
            results_refine = results.pandas().xyxy[0].values
            nms_human = len(results_refine)

            # Prepare the JSON response
            response_data = {
                'nms_human': nms_human,
            }
            
            # Convert to JSON
            json_response = json.dumps(response_data)

            # Yield the JSON response
            yield f"data: {json_response}\n\n"
            # print(f"{json_data_generator}")

            # await asyncio.sleep(0.1)
            

    return StreamingResponse(json_data_generator(), media_type="text/event-stream")
    

