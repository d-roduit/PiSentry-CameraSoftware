import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import streaming, recordings, thumbnails, config
from picamera import picam

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

api.include_router(streaming.router)
api.include_router(recordings.router)
api.include_router(thumbnails.router)
api.include_router(config.router)

if __name__ == '__main__':
    picam.start_detection()

    async def main():
        uvicorn_config = uvicorn.Config('main:api', host='0.0.0.0', port=9090, reload=True)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()
    asyncio.run(main())