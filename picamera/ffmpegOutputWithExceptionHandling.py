from picamera2.outputs import FfmpegOutput

class FfmpegOutputWithExceptionHandling(FfmpegOutput):
    def stop(self):
        try:
            super().stop()
        except Exception as e:
            print('Exception caught while stopping FfmpegOutput:', e)

    def outputframe(self, frame, keyframe=True, timestamp=None):
        try:
            super().outputframe(frame, keyframe, timestamp)
        except Exception as e:
            print('Exception caught in FfmpegOutput while outputting frame:', e)
            self.stop()
