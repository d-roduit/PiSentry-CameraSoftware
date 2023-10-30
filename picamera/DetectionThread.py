import threading
import os
import shutil
import datetime
import time
import cv2
import requests
from typing import Any
from picamera.motionDetection import MotionDetector
from picamera.objectDetection import ObjectDetector
from picamera.detection_typing import BoundingBox
from ConfigManager import configManager
from urls import backend_api_url

# DEBUG VARIABLES
debug_display_video_window = False
debug_draw_detection_boxes_on_video = False

def time_in_range(time_to_check, start_time, end_time) -> bool:
    """Return true if time_to_check is in the range [start_time, end_time]"""
    if not isinstance(time_to_check, datetime.time) or not isinstance(start_time, datetime.time) or not isinstance(end_time, datetime.time):
        raise ValueError('All function parameters must be of type `datetime.time`')

    if start_time <= end_time:
        return start_time <= time_to_check <= end_time
    else:
        return start_time <= time_to_check or time_to_check <= end_time

def draw_on_frame(frame, bounding_box, bounding_box_color=(0, 255, 0), object_type=None, confidence=None,
                  text_color=(0, 125, 0)) -> None:
    x, y, width, height = bounding_box
    top_left_corner = (x, y)
    bottom_right_corner = (x + width, y + height)
    cv2.rectangle(img=frame, pt1=top_left_corner, pt2=bottom_right_corner, color=bounding_box_color,
                  thickness=2)

    if object_type is not None:
        cv2.putText(frame, object_type.upper(), (top_left_corner[0] + 10, top_left_corner[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, text_color, 2)

    if confidence is not None:
        cv2.putText(frame, str(round(confidence * 100, 2)), (top_left_corner[0] + 200, top_left_corner[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, text_color, 2)

def get_object_with_highest_weight_and_area(all_detections, objects_to_detect_with_weights) -> tuple:
    output = ()

    def bounding_box_area(bounding_box: BoundingBox) -> int:
        x, y, width, height = bounding_box
        return height * width

    for detections_in_sub_frame in all_detections:
        for object_to_check in detections_in_sub_frame:
            if len(output) == 0:
                output = object_to_check
                continue

            object_to_check_type, _, object_to_check_bounding_box = object_to_check
            output_type, _, output_bounding_box = output

            object_to_check_weight = objects_to_detect_with_weights.get(object_to_check_type)
            output_weight = objects_to_detect_with_weights.get(output_type)

            if object_to_check_weight > output_weight:
                output = object_to_check
                continue

            if object_to_check_weight == output_weight:
                object_to_check_bounding_box_area = bounding_box_area(object_to_check_bounding_box)
                output_bounding_box_area = bounding_box_area(output_bounding_box)

                if object_to_check_bounding_box_area > output_bounding_box_area:
                    output = object_to_check

    return output

def write_recording_thumbnail_to_file(frame, filename, file_extension):
    if not os.path.isabs(configManager.config.detection.recordingsFolderPath):
        raise ValueError(
            'The recordings folder path must be an absolute path. Received:',
            configManager.config.detection.recordingsFolderPath
        )

    recordings_thumbnails_folderpath = os.path.join(
        configManager.config.detection.recordingsFolderPath,
        'thumbnails',
    )

    os.makedirs(name=recordings_thumbnails_folderpath, exist_ok=True)

    recording_thumbnail_filename = f'{filename}{file_extension}'
    recording_thumbnail_filepath = os.path.join(recordings_thumbnails_folderpath, recording_thumbnail_filename)

    cv2.imwrite(recording_thumbnail_filepath, frame, [cv2.IMWRITE_WEBP_QUALITY, 50])


def extract_square_thumbnail(frame, object_bounding_box):
    # Computer square side size
    object_box_x, object_box_y, object_box_width, object_box_height = object_bounding_box

    object_box_biggest_side_size = object_box_width if object_box_width > object_box_height else object_box_height
    square_side_size = object_box_biggest_side_size + 20 # padding of 10px on each side to fully encompass the object

    # Make sure that square is not bigger than frame width / height itself
    frame_width = len(frame[0]) - 1
    frame_height = len(frame) - 1

    square_side_size = square_side_size if square_side_size <= frame_width else frame_width
    square_side_size = square_side_size if square_side_size <= frame_height else frame_height

    # Place square in frame
    object_box_center_x = object_box_x + object_box_width // 2
    object_box_center_y = object_box_y + object_box_height // 2
    square_side_half_size = square_side_size // 2
    top_left_x = object_box_center_x - square_side_half_size
    top_left_y = object_box_center_y - square_side_half_size
    bottom_right_x = object_box_center_x + square_side_half_size
    bottom_right_y = object_box_center_y + square_side_half_size

    if top_left_x < 0: # if exceeds before the frame on the left
        top_left_x = 0
        bottom_right_x = square_side_size
    elif bottom_right_x > frame_width: # if exceeds beyond the frame on the right
        bottom_right_x = frame_width
        top_left_x = frame_width - square_side_size

    if top_left_y < 0: # if exceeds above the frame at the top
        top_left_y = 0
        bottom_right_y = square_side_size
    elif bottom_right_y > frame_height: # if exceeds below the frame at the bottom
        bottom_right_y = frame_height
        top_left_y = frame_height - square_side_size

    return frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]


def bytes_to_mebibytes(nb_bytes: int) -> float:
    nb_bytes_in_one_mebibyte = 1048576  # = 1024^2 OR 2^20
    return nb_bytes / nb_bytes_in_one_mebibyte

class DetectionThread(threading.Thread):
    def __init__(self, picam):
        threading.Thread.__init__(self)
        self._running = True
        self._picam = picam
        self._http_session = requests.Session()
        self._http_session.headers = { 'Authorization': configManager.config.user.token }

        self._configuration: dict[str, Any] = {
            'recording_phase_duration_in_seconds': 8,
            'max_recording_phase_duration_in_seconds': 120,
            'checking_phase_duration_in_seconds': 60,
        }

        self._motion_detector = MotionDetector()
        self._motion_detector.configure_detection(
            delta_threshold=9,
            min_area_for_motion=1000,
            min_frames_for_motion=13
        )

        directory_path = os.path.dirname(__file__)
        object_detection_directory = os.path.join(directory_path, 'objectDetection')
        coco_dataset_file_path = os.path.join(object_detection_directory, 'coco.names')
        neural_network_weights_file_path = os.path.join(object_detection_directory, 'frozen_inference_graph.pb')
        neural_network_config_file_path = os.path.join(object_detection_directory,
                                                       'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')

        self._object_detector = ObjectDetector()
        self._object_detector.configure_dataset(coco_dataset_file_path)
        self._object_detector.configure_neural_network(
            neural_network_weights_file_path=neural_network_weights_file_path,
            neural_network_config_file_path=neural_network_config_file_path,
            input_size=(320, 320),
            input_scale=1.0 / 127.5,
            input_mean=(127.5, 127.5, 127.5),
            input_swap_rb=True
        )
        self._object_detector.configure_detection(
            confidence_threshold=0.65,
            non_maximum_suppression_threshold=0.1,
            objects_to_detect=list(configManager.config.detection.objectTypes.keys()),
            min_frames_for_detection=5
        )

        try:
            self._detection_start_time = datetime.datetime.strptime(configManager.config.detection.startTime, '%H:%M').time()
            self._detection_end_time = datetime.datetime.strptime(configManager.config.detection.endTime, '%H:%M').time()
        except:
            raise ValueError('start time and end time strings must be formatted as follows: hours must have two digits and go from 00 to 23, minutes must have two digits and go from 00 to 59.') from None

    def run(self):
        try:
            while self._running:
                self._wait_for_detection_time_range()
                self._run_detection() # run the motion and object detection code
        except BaseException:
            self.stop()


    def _wait_for_detection_time_range(self):
        print('entering _wait_for_detection_time_range()')
        current_time = datetime.datetime.now().time()
        print('current_time:', current_time)

        while not time_in_range(current_time, self._detection_start_time, self._detection_end_time):
            print('starting to sleep...')
            seconds_to_wait_before_next_check = 10
            time.sleep(seconds_to_wait_before_next_check)
            current_time = datetime.datetime.now().time()
            print('after sleeping... current_time now:', current_time)

        print('leaving _wait_for_detection_time_range()')

    def _run_detection(self):
        print('entering _run_detection()')

        if debug_display_video_window:
            cv2.startWindowThread()
            cv2.namedWindow('debug', cv2.WINDOW_NORMAL)

        self._motion_detector.reset_detector()
        self._object_detector.reset_detector()

        previous_object_type_detected = None
        time_when_object_was_detected = time.time()
        time_when_recording_phase_started = time.time()
        time_when_checking_phase_started = time.time()
        is_in_recording_phase = False  # Recording video of detection
        is_in_checking_phase = False  # Checking if the last object detected is still there

        # Grab the array representing the frame
        while (frame := self._picam.get_frame()) is not None:
            if not is_in_recording_phase:
                current_time = datetime.datetime.now().time()
                print('checking if time_in_range. current_time', current_time)

                if not time_in_range(current_time, self._detection_start_time, self._detection_end_time):
                    print('\nleaving _run_detection(). current_time:', current_time)
                    break

            if is_in_recording_phase:
                current_time = time.time()
                seconds_elapsed_since_object_was_detected = current_time - time_when_object_was_detected
                seconds_elapsed_since_recording_phase_started = current_time - time_when_recording_phase_started

                if (
                        seconds_elapsed_since_object_was_detected >= self._configuration['recording_phase_duration_in_seconds']
                        or seconds_elapsed_since_recording_phase_started >= self._configuration['max_recording_phase_duration_in_seconds']
                ):
                    self._picam.stop_recording()
                    print('stop recording')
                    is_in_recording_phase = False
                    is_in_checking_phase = True
                    time_when_checking_phase_started = time.time()
                    print('-------------- leaving recording phase')
                    print('-------------- entering checking phase')

            elif is_in_checking_phase:
                current_time = time.time()
                seconds_elapsed_since_checking_phase_started = current_time - time_when_checking_phase_started

                if seconds_elapsed_since_checking_phase_started >= self._configuration['checking_phase_duration_in_seconds']:
                    is_in_checking_phase = False
                    print('-------------- leaving checking phase')
                    previous_object_type_detected = None

            motion_detection = self._motion_detector.detect(frame=frame, detection_areas=configManager.config.detection.areas)

            if debug_draw_detection_boxes_on_video:
                for motion_detection_area in configManager.config.detection.areas:
                    draw_on_frame(frame, motion_detection_area, (200, 100, 0))

            if len(motion_detection) == 0:
                if debug_display_video_window:
                    cv2.imshow('debug', frame)
                continue

            movement_bounding_boxes, _ = motion_detection

            print('number of movement bounding boxes:', len(movement_bounding_boxes))

            print('\n--------- Object ---------\n')

            object_detection: list[list[tuple]] = self._object_detector.detect(frame=frame,
                                                                               interest_areas=movement_bounding_boxes)

            if debug_draw_detection_boxes_on_video:
                for movement_bounding_box in movement_bounding_boxes:
                    draw_on_frame(frame, movement_bounding_box)

            if len(object_detection) == 0:
                if debug_display_video_window:
                    cv2.imshow('debug', frame)
                continue

            for object_detections_in_sub_frame in object_detection:
                print('\n--------- Object detection in sub frame\n')

                print('nb objects detected in sub frame:', len(object_detections_in_sub_frame))

                if len(object_detections_in_sub_frame) == 0:
                    continue

                for detection_result_of_one_object in object_detections_in_sub_frame:
                    object_type, object_confidence, object_bounding_box = detection_result_of_one_object

                    if debug_draw_detection_boxes_on_video:
                        draw_on_frame(frame, object_bounding_box, (0, 125, 0), object_type, object_confidence)

            object_to_notify = get_object_with_highest_weight_and_area(object_detection,
                                                                       configManager.config.detection.objectTypes)

            print('object_with_highest_weight_and_area:', object_to_notify)

            if len(object_to_notify) == 0:
                if debug_display_video_window:
                    cv2.imshow('debug', frame)
                continue

            object_to_notify_type, object_to_notify_confidence, object_to_notify_bounding_box = object_to_notify

            if debug_draw_detection_boxes_on_video:
                draw_on_frame(frame, object_to_notify_bounding_box, (0, 100, 200), object_to_notify_type,
                              object_to_notify_confidence, (0, 100, 200))

            print(object_to_notify_type, 'DETECTED')
            print('previous_object_type_detected:', previous_object_type_detected)
            print('is_in_recording_phase:', is_in_recording_phase)
            print('is_in_checking_phase:', is_in_checking_phase)

            must_notify_detection = (
                    (not is_in_recording_phase and not is_in_checking_phase)
                    or (is_in_checking_phase and object_to_notify_type != previous_object_type_detected)
            )

            if must_notify_detection:
                is_in_checking_phase = False
                is_in_recording_phase = True
                time_when_recording_phase_started = time.time()
                print('-------------- entering recording phase')

                print('start recording')
                recording_datetime = datetime.datetime.now()
                recording_filename = f'recording_{recording_datetime.strftime("%d-%m-%Y_%H-%M-%S")}'
                self._picam.start_recording(recording_filename)

                threading.Thread(
                    target=self.save_recording_and_notify_user,
                    args=(
                        frame,
                        object_to_notify_bounding_box,
                        recording_datetime,
                        recording_filename,
                        object_to_notify_type
                    )
                ).start()

            elif is_in_checking_phase and object_to_notify_type == previous_object_type_detected:
                time_when_checking_phase_started = time.time()

            time_when_object_was_detected = time.time()
            previous_object_type_detected = object_to_notify_type

            if debug_display_video_window:
                cv2.imshow('debug', frame)

    def save_recording_and_notify_user(self, frame, object_bounding_box, recording_datetime, recording_filename, object_to_notify_type):
        recording_file_extension = '.mp4'
        thumbnail_file_extension = '.webp'
        try:
            thumbnail_subframe = extract_square_thumbnail(frame, object_bounding_box)
            write_recording_thumbnail_to_file(thumbnail_subframe, recording_filename, thumbnail_file_extension)
        except Exception as e:
            print('Exception caught. Could not create recording thumbnail. Exception:', e)

        try:
            create_detection_session_response = self._http_session.post(
                backend_api_url + '/v1/detection-sessions',
                timeout=5
            )
            create_detection_session_response.raise_for_status()
            detection_session_response_json_data = create_detection_session_response.json()
            detection_session_id = detection_session_response_json_data['session_id']

            create_recording_response = self._http_session.post(
                backend_api_url + '/v1/recordings',
                json={
                    'recorded_at': recording_datetime.isoformat(),
                    'recording_filename': recording_filename,
                    'recording_extension': recording_file_extension,
                    'thumbnail_filename': recording_filename,
                    'thumbnail_extension': thumbnail_file_extension,
                    'detected_object_type': object_to_notify_type,
                    'detection_session_id': detection_session_id,
                    'camera_id': configManager.config.camera.id,
                },
                timeout=5)
            create_recording_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print('Request exception caught. Could not create detection session and recording. Exception:', e)

        # SEND NOTIFICATION OF DETECTION
        if configManager.config.notification.enabled:
            print('SENDING NOTIFICATION OF DETECTION FOR OJBECT', object_to_notify_type)

        # Keep enough free space on disk to write a future recording
        self.ensure_free_space(
            free_space_to_ensure_mebibytes = 400,
            filenames_to_keep = [f'{recording_filename}{recording_file_extension}']
        )

    def ensure_free_space(self, free_space_to_ensure_mebibytes, filenames_to_keep):
        recordings_folder_path = configManager.config.detection.recordingsFolderPath
        recording_file_extension = '.mp4'
        thumbnail_file_extension = '.webp'

        filenames_in_dir = (
            filename
            for filename in os.listdir(recordings_folder_path)
            if filename.endswith(recording_file_extension)
               and os.path.isfile(os.path.join(recordings_folder_path, filename))
               and filename not in filenames_to_keep
        )

        sorted_recordings_filenames = sorted(
            filenames_in_dir,
            key=lambda filename: os.path.getctime(os.path.join(recordings_folder_path, filename))
        )  # Oldest file first, newest file last

        _, _, current_free_space = shutil.disk_usage(recordings_folder_path)

        current_free_space_mebibytes = bytes_to_mebibytes(current_free_space)

        while current_free_space_mebibytes < free_space_to_ensure_mebibytes and len(sorted_recordings_filenames) > 0:
            # Create paths to files
            recording_filename = sorted_recordings_filenames.pop(0)
            recording_filename_without_extension = recording_filename.removesuffix(recording_file_extension)
            thumbnail_filename = f'{recording_filename_without_extension}{thumbnail_file_extension}'

            recording_filepath = os.path.join(recordings_folder_path, recording_filename)
            thumbnail_filepath = os.path.join(recordings_folder_path, 'thumbnails', thumbnail_filename)

            try:
                # Remove video file from disk
                recording_size_mebibytes = bytes_to_mebibytes(os.path.getsize(recording_filepath))
                os.remove(recording_filepath)

                # Update free space value
                current_free_space_mebibytes += recording_size_mebibytes

                # This HTTP call is placed here (and not before the video deletion or after the thumbnail deletion)
                # because it only makes sense to execute this code if the video deletion was successful AND we want to execute
                # it whether the thumbnail deletion is successful or not.
                # It also has its own try...except block because even if it fails, we still want to try to delete the thumbnail.
                try:
                    # Remove video file entry in database
                    delete_recording_response = self._http_session.delete(
                        backend_api_url + '/v1/recordings',
                        json={'recording_filename': recording_filename_without_extension},
                        timeout=5,
                    )
                    delete_recording_response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print('Request exception caught. Could not delete recording in database. Exception:', e)

                # Remove thumbnail file from disk
                thumbnail_size_mebibytes = bytes_to_mebibytes(os.path.getsize(thumbnail_filepath))
                os.remove(thumbnail_filepath)

                # Update free space value
                current_free_space_mebibytes += thumbnail_size_mebibytes
            except Exception as e:
                print('Exception caught while freeing space. Exception:', e)


    def stop(self):
        self._http_session.close()
        self._running = False
