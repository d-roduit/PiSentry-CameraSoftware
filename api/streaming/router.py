from fastapi import APIRouter

from picamera import picam

router = APIRouter(
    prefix="/streaming",
    tags=["streaming"],
)

@router.post("/start")
def start_streaming():
    picam.start_streaming()

@router.post("/stop")
def stop_streaming():
    picam.stop_streaming()

