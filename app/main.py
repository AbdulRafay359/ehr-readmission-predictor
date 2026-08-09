# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config.settings import settings
from app.db.mongodb import connect_to_mongo
import os

app = FastAPI(title="EHR Readmission Predictor", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS if hasattr(settings, 'ALLOWED_ORIGINS') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup():
    connect_to_mongo()

@app.get("/")
async def root():
    return {"message": "EHR Readmission Predictor API", "version": "2.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok", "model_version": "2.0.0"}