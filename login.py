from fastapi import FastAPI, HTTPException, Depends, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession
from fastapi.staticfiles import StaticFiles

# MySQL 데이터베이스 연결 정보
DATABASE_URL = "mysql+aiomysql://root:root@localhost/yolo_db"

# FastAPI 애플리케이션 생성
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

# 데이터베이스 세션 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 비동기 데이터베이스 세션 생성
engine_async = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(bind=engine_async, class_=AsyncSession, expire_on_commit=False)

# 데이터베이스 연결 종료 시 세션을 닫도록 변경
def get_db():
    db = async_session()
    try:
        yield db
    finally:
        db.close()

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
