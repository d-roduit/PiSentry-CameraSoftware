<h1 align="center">
   PiSentry Camera Software
</h1>

## Table of Contents

1. [Run the software](#run-software)
2. [Technologies](#technologies)
3. [License](#license)

## <a name="run-software"></a>Run the software

#### 1. Set correct URLs

Because the software interacts with other PiSentry projects (the backend API and the media server), you might want to check their URLs in `urls.py` and adapt them if necessary.

#### 2. Set correct configuration values

The file `config.json` contains the default configuration of the camera.
It is important to set your own values for the existing settings for the project to work properly.

#### 4. Run the software

> [!NOTE]
> The project has been tested with Python 3.13.5

1. If you are using a virtual environment, activate it by executing `source /<path_to_your_venv>/bin/activate`.
2. Install the necessary dependencies with `pip install -r requirements.txt`.
<br>Picamera2 is not listed in the `requirements.txt` file as it is managed via `apt` and should already be installed as it comes pre-installed in all Raspberry Pi OS images as of mid-September 2022.
<br>This project uses Picamera2 `v0.3.33`, which comes pre-installed with Raspberry Pi OS (Debian Trixie). If you need to update your package, you can do it with `sudo apt install -y python3-picamera2`.
<br>If you encounter a system without Picamera2 pre-installed, please see the installation process in the [Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf).  
4. Then, go to the root of the project and run the main file: `python main.py` and voilà !

## <a name="technologies"></a>Technologies

The software consists of two parts:
- the camera software, developed with [Picamera2](https://github.com/raspberrypi/picamera2)
- an API, developed with [FastAPI](https://fastapi.tiangolo.com/)

The main libraries used are :
- [OpenCV](https://pypi.org/project/opencv-python/) - for object detection, image processing, etc.
- [NumPy](https://numpy.org/) - for image processing
- [Requests](https://pypi.org/project/requests/) - for fetching/sending data from/to the backend API
- [Uvicorn](https://www.uvicorn.org/) - for serving the API

## <a name="license"></a>License

This project is licensed under the MIT License
