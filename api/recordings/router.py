from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(
    prefix='/recordings',
    tags=['recordings'],
)

@router.get('/')
def get_recordings():
    return { 'recordings': ['mavideo.mp4', 'mavideo2.mp4', 'mavideo3.mp4'] }

@router.get('/{filename}', response_class=FileResponse)
def get_recording(filename: str):
    return '/home/droduit/Desktop/' + filename

