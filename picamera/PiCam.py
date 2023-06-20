from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput, CircularOutput
from datetime import datetime
import subprocess
import os

class PiCam:
    def __init__(self):
        self._picam2 = Picamera2()
        video_config = self._picam2.create_video_configuration(main={'format': 'RGB888', 'size': (1920, 1080)})
        self._picam2.configure(video_config)

        # self._streamingOutput = FfmpegOutput('-f flv rtmp://mediaserver.pisentry.app/PiSentry/Spooky_Stream')
        self._streamingOutput = FfmpegOutput('-f flv rtmp://192.168.1.211:1935/PiSentry/Spooky_Stream')
        self._recordingOutput = CircularOutput(buffersize=150)  # 150 means 30 images * 5 seconds
        self._encoder = H264Encoder(repeat=True, iperiod=15)
        # self._encoder.output = [self._streamingOutput, self._recordingOutput]
        self._encoder.output = [self._recordingOutput]

        self._picam2.start()
        self._picam2.start_encoder(self._encoder)


        self._recording_filepath = ''

    def start_streaming(self):
        try:
            # self._picam2.start_encoder(self._encoder, self._streamingOutput)
            # self._picam2.start_encoder(self._encoder)
            self._streamingOutput.start()
        except Exception as exception:
            print(f'Exception caught in start_streaming() : {exception}')

    def stop_streaming(self):
        try:
            # self._picam2.stop_encoder()
            self._streamingOutput.stop()
        except Exception as exception:
            print(f'Exception caught in stop_streaming() : {exception}')

    def start_recording(self):
        try:
            self._recording_filepath = f'recording_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
            self._recordingOutput.fileoutput = f'{self._recording_filepath}.h264'
            self._recordingOutput.start()
        except Exception as exception:
            print(f'Exception caught in start_recording() : {exception}')

    def stop_recording(self):
        try:
            self._recordingOutput.stop()

            try:
                subprocess.run(['ffmpeg', '-framerate', '30', '-i', f'{self._recording_filepath}.h264', '-c', 'copy',
                                f'{self._recording_filepath}.mp4'], check=True, timeout=60)
            except FileNotFoundError as file_not_found_exception:  # one of the program called does not exist
                print(f'Process failed because the executable could not be found.\n{file_not_found_exception}')
            except subprocess.CalledProcessError as called_process_exception:  # subprocess execution returned a non-zero code
                print(
                    f'Process execution did not return a successful return code (0). '
                    f'Returned {called_process_exception.returncode}\n{called_process_exception}'
                )
            except subprocess.TimeoutExpired as timeout_exception:  # program did not finish its task before timeout
                print(f'Process timed out.\n{timeout_exception}')

            if os.path.exists(f'{self._recording_filepath}.h264'):
                os.remove(f'{self._recording_filepath}.h264')
                print(f'File "{self._recording_filepath}.h264" removed successfully')
            else:
                print(f'The file "{self._recording_filepath}.h264" does not exist')
        except Exception as exception:
            print(f'Exception caught in stop_recording() : {exception}')

    def get_frame(self):
        return self._picam2.capture_array()

picam = PiCam()