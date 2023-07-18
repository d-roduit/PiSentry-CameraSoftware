from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput, CircularOutput
from picamera.DetectionThread import DetectionThread
from datetime import datetime
import subprocess
import os
from ConfigManager import configManager


def from_h264_create_file(h264_recording_filepath: str, new_file_extension: str):
    if not os.path.isfile(h264_recording_filepath):
        raise ValueError('The path must point to a file. Received:', h264_recording_filepath)

    h264_extension = '.h264'

    if not h264_recording_filepath.endswith(h264_extension):
        raise ValueError(f'A file path ending with extension {h264_extension} must be provided')

    if not new_file_extension.startswith('.'):
        raise ValueError('The new file extension must starts with a point (.)')

    recording_filepath_without_extension = h264_recording_filepath.removesuffix(h264_extension)
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
        self._recordingOutput = CircularOutput(buffersize=300)  # 300 means 30 images * 10 seconds
        self._encoder = H264Encoder(repeat=True, iperiod=15)

        self._picam2.start_encoder(self._encoder)
        self._picam2.start()

        # This line must be after the `start_encoder()` line not to start the outputs at the same time as the encoder
        self._encoder.output = [self._streamingOutput, self._recordingOutput]

        self._h264_recording_filepath = ''

        self._detection_thread = None

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
            if not os.path.isdir(configManager.config.detection.recordingsFolderPath):
                raise ValueError('The recordings folder path must point to a directory. Received:', configManager.config.detection.recordingsFolderPath)

            if not os.path.isabs(configManager.config.detection.recordingsFolderPath):
                raise ValueError('The recordings folder path must be an absolute path. Received:', configManager.config.detection.recordingsFolderPath)

            recording_filename = f'recording_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.h264'

            self._h264_recording_filepath = os.path.join(configManager.config.detection.recordingsFolderPath, recording_filename)

            self._recordingOutput.fileoutput = self._h264_recording_filepath
            self._recordingOutput.start()
        except Exception as exception:
            print(f'Exception caught in start_recording() : {exception}')

    def stop_recording(self):
        try:
            self._recordingOutput.stop()

            from_h264_create_file(self._h264_recording_filepath, '.mp4')

            if not os.path.exists(self._h264_recording_filepath):
                print(f'File "{self._h264_recording_filepath}" does not exist')
                return

            os.remove(self._h264_recording_filepath)
            print(f'File "{self._h264_recording_filepath}" removed successfully')
        except Exception as exception:
            print(f'Exception caught in stop_recording() : {exception}')

    def get_frame(self):
        return self._picam2.capture_array()

    def start_detection(self):
        if self._detection_thread is not None and self.is_detection_running:
            return
        self._detection_thread = DetectionThread(self)
        self._detection_thread.start()
    def stop_detection(self):
        if self._detection_thread is None or not self.is_detection_running:
            return
        self._detection_thread.stop()
        self._detection_thread.join()

    @property
    def is_detection_running(self):
        return self._detection_thread.is_alive() if self._detection_thread is not None else False

picam = PiCam()
