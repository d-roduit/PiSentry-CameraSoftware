import os
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from ConfigManager import configManager
from picamera import picam
import cv2
import time
import numpy as np

router = APIRouter(
    prefix='/thumbnails',
    tags=['thumbnails'],
)

live_thumbnail_creation_timestamp = None
live_thumbnail_uptodate_duration = 10 # in seconds

@router.get('/live', response_class=FileResponse)
def get_live_thumbnail():
    global live_thumbnail_creation_timestamp

    thumbnails_folderpath = os.path.join(configManager.config.detection.recordingsFolderPath, 'thumbnails')

    if not os.path.isdir(thumbnails_folderpath):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail='The thumbnails folder path must point to an existing directory')

    thumbnail_filename = 'live.webp'
    thumbnail_filepath = os.path.join(thumbnails_folderpath, thumbnail_filename)

    must_create_live_thumbnail = True

    # We check if the file exists because it could have been deleted by the DetectionThread
    # when ensuring to keep enough free space for recordings. If the file does not exist, we must create it anyway.
    if live_thumbnail_creation_timestamp is not None and os.path.isfile(thumbnail_filepath):
        current_timestamp = time.time()
        nb_seconds_since_live_thumbnail_creation = current_timestamp - live_thumbnail_creation_timestamp
        must_create_live_thumbnail = nb_seconds_since_live_thumbnail_creation > live_thumbnail_uptodate_duration

    if must_create_live_thumbnail:
        frame = picam.get_frame()
        height, width, _ = np.shape(frame)
        downscaled_dimensions = (width // 2, height // 2)
        resized_frame = cv2.resize(frame, downscaled_dimensions, interpolation=cv2.INTER_AREA)
        cv2.imwrite(thumbnail_filepath, resized_frame, [cv2.IMWRITE_WEBP_QUALITY, 50])
        live_thumbnail_creation_timestamp = time.time()

    return thumbnail_filepath


@router.get('/{thumbnail_filename}', response_class=FileResponse)
def get_thumbnail(thumbnail_filename: str):
    thumbnails_folderpath = os.path.join(configManager.config.detection.recordingsFolderPath, 'thumbnails')

    if not os.path.isdir(thumbnails_folderpath):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail='The thumbnails folder path must point to an existing directory')

    thumbnail_filepath = os.path.join(thumbnails_folderpath, thumbnail_filename)

    if not os.path.isfile(thumbnail_filepath):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Thumbnail not found')

    return thumbnail_filepath

