from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

class PiCam:
    def __init__(self):
        self.picam2 = Picamera2()
        video_config = self.picam2.create_video_configuration(
            main={'size': (1920, 1080)},
            lores={'size': (1920, 1080)}
        )
        self.picam2.configure(video_config)
        self.picam2.start()


    def start_streaming(self):
        encoder = H264Encoder()
        output = FfmpegOutput("-f flv rtmp://192.168.1.211:1935/PiSentry/Spooky_Stream")
        # output = FfmpegOutput("-f flv rtmp://mediaserver.pisentry.app/PiSentry/Spooky_Stream")
        try:
            self.picam2.start_encoder(encoder, output)
        except Exception as exception:
            print(f'Exception caught in start_streaming() : {exception}')

    def stop_streaming(self):
        try:
            self.picam2.stop_encoder()
        except Exception as exception:
            print(f'Exception caught in stop_streaming() : {exception}')

picam = PiCam()