from fastapi import FastAPI, HTTPException, Depends, Form, Request, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import cv2
import torch
import datetime
import json
import asyncio

app = FastAPI()

# CORS 설정
origins = [
    "http://localhost:3000/Regist",
    "http://localhost:3000/Login",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 데이터베이스 연결 설정
DATABASE_URL = "mysql+aiomysql://root:root@localhost/yolo_db"
engine: AsyncEngine = create_async_engine(DATABASE_URL, echo=True, future=True)
metadata = MetaData()

# 테이블 정의
users = Table(
    'users', metadata,
    Column('index_id', Integer, primary_key=True, autoincrement=True),
    Column('id', String(255), primary_key=True),
    Column('username', String(255), unique=True, index=True),
    Column('password', String(255)),
    Column('phone_number', String(20)),
)

# 테이블 생성
# metadata.create_all(engine)

# 비동기 데이터베이스 세션 생성
engine_async = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(bind=engine_async, class_=AsyncSession, expire_on_commit=False)

# 데이터베이스 세션 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_async)


# 데이터베이스 연결 종료 시 세션을 닫도록 변경
def get_db():
    db = async_session()
    try:
        yield db
    finally:
        db.close()

# 세션 사용을 위한 의존성 함수
def get_current_user(request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if user_id:
        db_user = db.execute(users.select().where(users.c.id == user_id)).fetchone()
        return db_user
    return None

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/login", response_class=HTMLResponse)
async def stream(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# 로그인 화면 렌더링 및 처리
@app.post("/userlogin", response_class=HTMLResponse)
async def login(id: str = Form(...), pwd: str = Form(...), db: AsyncSession = Depends(get_db)):
    query = users.select().where(users.c.id == id)
    db_user = await db.execute(query)
    db_user = db_user.fetchone()

    print(f"Found user: {db_user}")
    if db_user:
        if db_user.password == pwd:
            return RedirectResponse(url="/login", status_code=303)
        else:
            raise HTTPException(status_code=400, detail="Incorrect password")
    else:
        raise HTTPException(status_code=400, detail="User not found")

# 회원가입 화면 렌더링
@app.get("/register", response_class=HTMLResponse)
async def render_register(request: Request):
    return templates.TemplateResponse("regist.html", {"request": request})

# 회원가입 처리
@app.post("/register")
async def process_register(
    id: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    phone_number: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    try:
        query_create_user = users.insert().values(id=id, username=username, password=password, phone_number=phone_number)
        await db.execute(query_create_user)
        await db.commit()
        return {"message": "Registration successful"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# YOLOv5 모델 불러오기
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l',
                            device='cuda:0' if torch.cuda.is_available() else 'cpu')  # 예측 모델
yolo_model.classes = [0]  # 예측 클래스 (index의 값 0번이 사람)

# 웹캠 비디오 캡처 설정 - 기본 웹캠
cap = cv2.VideoCapture(0)

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
            
            await asyncio.sleep(0.1)

    return StreamingResponse(json_data_generator(), media_type="text/event-stream")
