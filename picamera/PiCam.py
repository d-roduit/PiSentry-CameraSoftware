from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput, CircularOutput
from datetime import datetime
import subprocess
import os

def from_h264_create_file(h264_recording_filepath: str, new_file_extension: str):
    h264_extension = '.h264'

    if not h264_recording_filepath.endswith(h264_extension):
        raise ValueError(f'A file path ending with extension {h264_extension} must be provided')

    if not new_file_extension.startswith('.'):
        raise ValueError('The new file extension must starts with a point (.)')

    recording_filepath_without_extension = h264_recording_filepath[:-len(h264_extension)]
    new_filepath = recording_filepath_without_extension + new_file_extension

    try:
        subprocess.run(['ffmpeg', '-framerate', '30', '-i', h264_recording_filepath, '-c', 'copy',
                        f'{new_filepath}'], check=True, timeout=60)
    except FileNotFoundError as file_not_found_exception:  # one of the program called does not exist
        print(f'Process failed because the executable could not be found.\n{file_not_found_exception}')
    except subprocess.CalledProcessError as called_process_exception:  # subprocess execution returned a non-zero code
        print(
            f'Process execution did not return a successful return code (0). '
            f'Returned {called_process_exception.returncode}\n{called_process_exception}'
        )
    except subprocess.TimeoutExpired as timeout_exception:  # program did not finish its task before timeout
        print(f'Process timed out.\n{timeout_exception}')

class PiCam:
    def __init__(self):
        self._picam2 = Picamera2()
        video_config = self._picam2.create_video_configuration(main={'format': 'RGB888', 'size': (1920, 1080)})
        self._picam2.configure(video_config)

        # self._streamingOutput = FfmpegOutput('-f flv rtmp://mediaserver.pisentry.app/PiSentry/Spooky_Stream')
        self._streamingOutput = FfmpegOutput('-f flv rtmp://192.168.1.211:1935/PiSentry/Spooky_Stream')
        self._recordingOutput = CircularOutput(buffersize=150)  # 150 means 30 images * 5 seconds
        self._encoder = H264Encoder(repeat=True, iperiod=15)

        self._picam2.start_encoder(self._encoder)
        self._picam2.start()

        # This line must be after the `start_encoder()` line not to start the outputs at the same time as the encoder
        self._encoder.output = [self._streamingOutput, self._recordingOutput]

        self._h264_recording_filepath = ''

    def start_streaming(self):
        try:
            self._streamingOutput.start()
        except Exception as exception:
            print(f'Exception caught in start_streaming() : {exception}')

    def stop_streaming(self):
        try:
            self._streamingOutput.stop()
        except Exception as exception:
            print(f'Exception caught in stop_streaming() : {exception}')

    def start_recording(self):
        try:
            self._h264_recording_filepath = f'./recording_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.h264'
            self._recordingOutput.fileoutput = self._h264_recording_filepath
            self._recordingOutput.start()
        except Exception as exception:
            print(f'Exception caught in start_recording() : {exception}')

    def stop_recording(self):
        try:
            self._recordingOutput.stop()

            from_h264_create_file(self._h264_recording_filepath, '.mp4')

            if os.path.exists(self._h264_recording_filepath):
                os.remove(self._h264_recording_filepath)
                print(f'File "{self._h264_recording_filepath}" removed successfully')
            else:
                print(f'File "{self._h264_recording_filepath}" does not exist')
        except Exception as exception:
            print(f'Exception caught in stop_recording() : {exception}')

    def get_frame(self):
        return self._picam2.capture_array()


picam = PiCam()
