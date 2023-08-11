from fastapi import APIRouter
from fastapi.responses import FileResponse
from picamera import picam

router = APIRouter(
    prefix='/recordings',
    tags=['recordings'],
)

@router.get('/{filename}', response_class=FileResponse)
def get_recording(filename: str):
    return '/home/droduit/Desktop/' + filename

@router.post('/start')
def start_recording():
    picam.start_recording()

