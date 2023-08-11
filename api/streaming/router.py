from fastapi import APIRouter, HTTPException, Response, status
from picamera import picam

router = APIRouter(
    prefix='/streaming',
    tags=['streaming'],
)

@router.post('/start')
def start_streaming():
    try:
        picam.start_streaming()
        return Response(status_code=status.HTTP_200_OK)
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Could not start streaming')

@router.post('/stop')
def stop_streaming():
    try:
        picam.stop_streaming()
        return Response(status_code=status.HTTP_200_OK)
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Could not stop streaming')

