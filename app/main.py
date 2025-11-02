from fastapi import FastAPI
from app.routers.recommend import router as recommend_router
from app import config
from app.services.recommender import load_model

app = FastAPI(title="Hrr Model API", version=config.MODEL_VERSION)

@app.on_event("startup")
def on_startup():
    # 프로세스 시작 시 모델 로드
    load_model()

app.include_router(recommend_router, prefix="")
