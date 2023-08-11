import os
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from picamera import picam
from ConfigManager import configManager

router = APIRouter(
    prefix='/recordings',
    tags=['recordings'],
)

@router.get('/{recording_filename}', response_class=FileResponse)
def get_recording(recording_filename: str):
    if not os.path.isdir(configManager.config.detection.recordingsFolderPath):
        raise HTTPException(status_code=500, detail='The recordings folder path must point to a directory')

    recording_filepath = os.path.join(configManager.config.detection.recordingsFolderPath, recording_filename)

    if not os.path.isfile(recording_filepath):
        return HTTPException(status_code=404, detail='Recording not found')

    return recording_filepath

@router.post('/start')
def start_recording():
    picam.start_recording()

