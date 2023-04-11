import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import streaming, recordings

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

if __name__ == '__main__':
    async def main():
        config = uvicorn.Config('main:api', host='0.0.0.0', port=9090, reload=True)
        server = uvicorn.Server(config)
        await server.serve()
    asyncio.run(main())