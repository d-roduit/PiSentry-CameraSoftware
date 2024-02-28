from fastapi import APIRouter, HTTPException, Response, status
from ConfigManager import configManager
import requests

router = APIRouter(
    prefix='/config',
    tags=['config'],
)

@router.post('/reload')
def reload_config_from_db():
    try:
        configManager.load_config_from_db()
    except (requests.exceptions.HTTPError, ValueError, KeyError) as error:
        raise HTTPException(status_code=status.HTTP_424_FAILED_DEPENDENCY, detail=error)

    return Response(status_code=status.HTTP_200_OK)
