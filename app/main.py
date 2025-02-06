import sys
import os
from fastapi import FastAPI

# Ensure 'app' is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from api.routes import router

app = FastAPI(
    title="Virtual Gown Try-On API",
    description="Backend for processing gown try-on with UNet, OpenPose, and CycleGAN models.",
    version="1.0.0",
)

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Virtual Gown Try-On API"}
