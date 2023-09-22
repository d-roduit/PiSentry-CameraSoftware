import os
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from ConfigManager import configManager

router = APIRouter(
    prefix='/thumbnails',
    tags=['thumbnails'],
)

@router.get('/{thumbnail_filename}', response_class=FileResponse)
def get_recording(thumbnail_filename: str):
    thumbnails_folderpath = os.path.join(configManager.config.detection.recordingsFolderPath, 'thumbnails')

    if not os.path.isdir(thumbnails_folderpath):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='The thumbnails folder path must point to an existing directory')

    thumbnail_filepath = os.path.join(thumbnails_folderpath, thumbnail_filename)

    if not os.path.isfile(thumbnail_filepath):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Thumbnail not found')

    return thumbnail_filepath

