from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput, FfmpegOutput
from picamera.DetectionThread import DetectionThread
import subprocess
import os
from ConfigManager import configManager
from urls import mediaserver_publish_livestream_url


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

    subprocess.run([
        'ffmpeg', '-framerate', '30', '-i', h264_recording_filepath, '-c', 'copy',
        '-movflags', 'faststart', new_filepath
    ], check=True, timeout=60)


class PiCam:
    def __init__(self):
        self._picam2 = Picamera2()
        video_config = self._picam2.create_video_configuration(main={'format': 'RGB888', 'size': (1920, 1080)})
        self._picam2.configure(video_config)

        camera_port = configManager.config.camera.port
        # self._streamingOutput = FfmpegOutput(f'-f flv rtmp://mediaserver.pisentry.app/pisentry/{camera_port}')
        self._streamingOutput = FfmpegOutput(f'-f flv {mediaserver_publish_livestream_url}/pisentry/{camera_port}')
        self._streamingOutput.error_callback = self.handle_ffmpeg_streaming_brokenpipe_exception
        self._recordingOutput = CircularOutput(buffersize=300)  # 300 means 30 images/second * 10 seconds
        self._encoder = H264Encoder(repeat=True, iperiod=15)

        self._picam2.start_encoder(self._encoder)
        self._picam2.start()

        # This line must be after the `start_encoder()` line not to start the outputs at the same time as the encoder
        self._encoder.output = [self._streamingOutput, self._recordingOutput]

        self._h264_recording_filepath = ''

        self._detection_thread = None

    def handle_ffmpeg_streaming_brokenpipe_exception(self, exception):
        # We call the `.stop()` method of the streaming output to make sure to benefit from future
        # improvements and changes in the upstream, meaning official picamera2, code of the `FfmpegOutput` class.
        # But in reality, we only need to set `self._streamingOutput.recording = False` to fix the brokenpipe exception
        # preventing to restart streaming. That's actually what the `.stop()` method of the `Output` base class does.
        self._streamingOutput.stop()

    def start_streaming(self):
        if not self._streamingOutput.recording:
            self.stop_streaming()
            self._streamingOutput.start()

    def stop_streaming(self):
        self._streamingOutput.stop()

    def start_recording(self, recording_filename):
        if not os.path.isabs(configManager.config.detection.recordingsFolderPath):
            raise ValueError('The recordings folder path must be an absolute path. Received:',
                             configManager.config.detection.recordingsFolderPath)

        os.makedirs(name=configManager.config.detection.recordingsFolderPath, exist_ok=True)

        file_extension = '.h264'

        self._h264_recording_filepath = os.path.join(configManager.config.detection.recordingsFolderPath,
                                                     f'{recording_filename}{file_extension}')

        self._recordingOutput.fileoutput = self._h264_recording_filepath
        self._recordingOutput.start()

    def stop_recording(self):
        self._recordingOutput.stop()

        from_h264_create_file(self._h264_recording_filepath, '.mp4')

        if not os.path.exists(self._h264_recording_filepath):
            print(f'File "{self._h264_recording_filepath}" does not exist')
            return

        os.remove(self._h264_recording_filepath)
        print(f'File "{self._h264_recording_filepath}" removed successfully')

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
