from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="Email Spam Detector",
    description="SMS spam/ham classification with MLflow-backed training.",
)

app.include_router(router)
