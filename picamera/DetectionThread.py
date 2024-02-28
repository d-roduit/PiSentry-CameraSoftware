import threading
import os
import shutil
import datetime
import time
import cv2
import requests

from typing import Any
from Observable import Observable
from Observer import Observer
from picamera.motionDetection import MotionDetector
from picamera.objectDetection import ObjectDetector
from picamera.helpers.DetectionActions import DetectionActions
from picamera.helpers.utils_functions import (
    time_in_range,
    get_object_with_highest_weight_and_area,
    draw_on_frame,
    bytes_to_mebibytes,
    write_frame_to_file,
    extract_square_thumbnail,
)
from ConfigManager import ConfigManager, configManager
from urls import (
    notifications_api_endpoint,
    thumbnails_api_endpoint,
    recordings_api_endpoint,
    detection_sessions_api_endpoint,
)

# DEBUG VARIABLES
debug_display_video_window = False
debug_draw_detection_boxes_on_video = False

class DetectionThread(threading.Thread, Observer):
    def __init__(self, picam):
        threading.Thread.__init__(self)
        self._running = True
        self._picam = picam

        configManager.add_observer(self)

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
            objects_to_detect=list(configManager.config.detection.objects.keys()),
            min_frames_for_detection=5
        )

        self._initialize_detection_and_notifications_times()

        # List of recordings which could not be inserted previously due to network errors
        # or any other exception which could have happened.
        # We hope to be able to empty this list by successfully inserting in database the recordings it contains
        # the next time we have a network connection or that the backend API is reachable and that the
        # `save_recording_and_notify_user()`method is called.
        self._recordings_awaiting_insertion_in_db = dict()

        # List of recordings filenames which could not be deleted previously due to network errors
        # or any other exception which could have happened.
        # We hope to be able to empty this list by successfully deleting from database the filenames it contains
        # the next time we have a network connection or that the backend API is reachable and that the
        # `ensure_free_space()` method is called.
        self._recordings_filenames_without_extension_awaiting_deletion = set()

    def _initialize_detection_and_notifications_times(self) -> None:
        try:
            self._detection_start_time = datetime.datetime.strptime(configManager.config.detection.startTime, '%H:%M').time()
            self._detection_end_time = datetime.datetime.strptime(configManager.config.detection.endTime, '%H:%M').time()
            self._notifications_start_time = datetime.datetime.strptime(configManager.config.notifications.startTime,'%H:%M').time()
            self._notifications_end_time = datetime.datetime.strptime(configManager.config.notifications.endTime,'%H:%M').time()
        except:
            raise ValueError('start time and end time strings must be formatted as follows: hours must have two digits and go from 00 to 23, minutes must have two digits and go from 00 to 59.') from None

    def update(self, observable: Observable) -> None:
        if isinstance(observable, ConfigManager):
            self._initialize_detection_and_notifications_times()

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

            object_to_notify = get_object_with_highest_weight_and_area(
                object_detection,
                configManager.config.detection.objects
            )

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

            # Check if we must start recording based on the user setting for this object type
            must_start_recording_for_object_type = False
            try:
                detection_action_for_object_type = configManager.config.detection.objects[object_to_notify_type].action
                must_start_recording_for_object_type = DetectionActions.must_record(detection_action_for_object_type)
            except ValueError as e:
                print(f'Could not check if we must start recording based on the user setting for object type "{object_to_notify_type}". Exception:', e)

            must_start_recording = (
                    must_start_recording_for_object_type
                    and (
                        (not is_in_recording_phase and not is_in_checking_phase)
                        or (is_in_checking_phase and object_to_notify_type != previous_object_type_detected)
                    )
            )

            if must_start_recording:
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
        thumbnail_filename = recording_filename
        square_thumbnail_filename = f'{thumbnail_filename}_square'
        recordings_thumbnails_folderpath = os.path.join(
            configManager.config.detection.recordingsFolderPath,
            'thumbnails',
        )

        try:
            square_thumbnail_subframe = extract_square_thumbnail(frame, object_bounding_box)
            resized_square_thumbnail_subframe = cv2.resize(square_thumbnail_subframe, dsize=(128, 128)) # make 128px x 128px image
            write_frame_to_file(
                resized_square_thumbnail_subframe,
                recordings_thumbnails_folderpath,
                f'{square_thumbnail_filename}{thumbnail_file_extension}',
                [cv2.IMWRITE_WEBP_QUALITY, 50]
            ) # create square thumbnail
        except Exception as e:
            print('Exception caught. Could not create recording\'s square thumbnail. Exception:', e)

        try:
            resized_rectangle_thumbnail_frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5) # divide resolution by 2
            write_frame_to_file(
                resized_rectangle_thumbnail_frame,
                recordings_thumbnails_folderpath,
                f'{thumbnail_filename}{thumbnail_file_extension}',
                [cv2.IMWRITE_WEBP_QUALITY, 50]
            ) # create rectangle thumbnail
        except Exception as e:
            print('Exception caught. Could not create recording\'s rectangle thumbnail. Exception:', e)

        try:
            create_detection_session_response = self._http_session.post(detection_sessions_api_endpoint, timeout=5)
            create_detection_session_response.raise_for_status()
            detection_session_response_json_data = create_detection_session_response.json()
            detection_session_id = detection_session_response_json_data['session_id']

            create_recording_response = self._http_session.post(
                recordings_api_endpoint,
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
            self._recordings_awaiting_insertion_in_db[recording_filename] = {
                'recorded_at': recording_datetime.isoformat(),
                'recording_filename': recording_filename,
                'recording_extension': recording_file_extension,
                'thumbnail_filename': recording_filename,
                'thumbnail_extension': thumbnail_file_extension,
                'detected_object_type': object_to_notify_type,
                'camera_id': configManager.config.camera.id,
            }
            print('Recording added to list of recordings awaiting insertion.')

        # Try to insert in database the recordings which could not be inserted previously
        # due to network errors or any other exception which could have happened.
        for recording_filename_awaiting_insertion, recording_data_awaiting_insertion  in self._recordings_awaiting_insertion_in_db.copy().items():
            try:
                create_detection_session_response = self._http_session.post(detection_sessions_api_endpoint, timeout=5)
                create_detection_session_response.raise_for_status()
                detection_session_response_json_data = create_detection_session_response.json()
                detection_session_id = detection_session_response_json_data['session_id']

                create_recording_response = self._http_session.post(
                    recordings_api_endpoint,
                    json={
                        'recorded_at': recording_data_awaiting_insertion['recorded_at'],
                        'recording_filename': recording_data_awaiting_insertion['recording_filename'],
                        'recording_extension': recording_data_awaiting_insertion['recording_extension'],
                        'thumbnail_filename': recording_data_awaiting_insertion['thumbnail_filename'],
                        'thumbnail_extension': recording_data_awaiting_insertion['thumbnail_extension'],
                        'detected_object_type': recording_data_awaiting_insertion['detected_object_type'],
                        'detection_session_id': detection_session_id,
                        'camera_id': recording_data_awaiting_insertion['camera_id'],
                    },
                    timeout=5)
                create_recording_response.raise_for_status()
                del self._recordings_awaiting_insertion_in_db[recording_filename_awaiting_insertion]
            except requests.exceptions.RequestException as e:
                print('Request exception caught. Could not create detection session and recording for awaiting recording. Exception:', e)

        # SEND NOTIFICATION OF DETECTION
        notification_current_time = datetime.datetime.now().time()

        # Check if a notification needs to be sent based on the user setting for this object type
        must_send_notification_for_object_type = False
        try:
            detection_action_for_object_type = configManager.config.detection.objects[object_to_notify_type].action
            must_send_notification_for_object_type = DetectionActions.must_send_notification(detection_action_for_object_type)
        except ValueError as e:
            print(f'Could not check if a notification needs to be sent based on the user setting for object type "{object_to_notify_type}". Exception:', e)

        must_notify_detection = (
            must_send_notification_for_object_type
            and configManager.config.notifications.enabled
            and time_in_range(notification_current_time, self._notifications_start_time, self._notifications_end_time)
        )

        if must_notify_detection:
            print('SENDING NOTIFICATION OF DETECTION FOR OJBECT', object_to_notify_type)

            camera_name = configManager.config.camera.name
            camera_id = configManager.config.camera.id
            user_token = configManager.config.user.token
            square_thumbnail_file = f'{square_thumbnail_filename}{thumbnail_file_extension}'

            icon_url = f'{thumbnails_api_endpoint}/{camera_id}/{square_thumbnail_file}?access_token={user_token}'

            try:
                send_notifications_response = self._http_session.post(
                    f'{notifications_api_endpoint}/{camera_id}/send',
                    json={
                        'notification': {
                            'title': camera_name,
                            'message': f'{object_to_notify_type.capitalize()} detected',
                            'icon': icon_url,
                            'timestamp': int(time.time() * 1000), # timestamp must be in milliseconds
                            'topic': 'detection',
                        }
                    },
                    timeout=10
                )
                send_notifications_response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print('Request exception caught. Could not send notification. Exception:', e)

        # Keep enough free space on disk to write a future recording
        self.ensure_free_space(
            free_space_to_ensure_mebibytes = 400,
            filenames_to_keep = [f'{recording_filename}{recording_file_extension}']
        )

    def ensure_free_space(self, free_space_to_ensure_mebibytes, filenames_to_keep):
        recordings_folder_path = configManager.config.detection.recordingsFolderPath
        recording_file_extension = '.mp4'
        thumbnail_file_extension = '.webp'
        thumbnail_formats = ('', '_square')

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

        # Try to delete from the database the recording filenames which could not be deleted previously,
        # due to network errors or any other exception which could have happened.
        for recording_filename_without_extension in self._recordings_filenames_without_extension_awaiting_deletion.copy():
            try:
                # Delete recording file entry in database
                delete_recording_response = self._http_session.delete(
                    recordings_api_endpoint,
                    json={'recording_filename': recording_filename_without_extension},
                    timeout=5,
                )
                delete_recording_response.raise_for_status()
                self._recordings_filenames_without_extension_awaiting_deletion.remove(recording_filename_without_extension)
            except requests.exceptions.RequestException as e:
                print('Request exception caught. Could not delete recording in database. Exception:', e)
            except KeyError as key_error:
                print('Could not remove filename from list of recordings filenames awaiting to be deleted from database. Exception:', key_error)

        while current_free_space_mebibytes < free_space_to_ensure_mebibytes and len(sorted_recordings_filenames) > 0:
            # Create paths to files
            recording_filename = sorted_recordings_filenames.pop(0)
            recording_filename_without_extension = recording_filename.removesuffix(recording_file_extension)

            recording_filepath = os.path.join(recordings_folder_path, recording_filename)

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
                        recordings_api_endpoint,
                        json={'recording_filename': recording_filename_without_extension},
                        timeout=5,
                    )
                    delete_recording_response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print('Request exception caught. Could not delete recording in database. Exception:', e)
                    self._recordings_filenames_without_extension_awaiting_deletion.add(recording_filename_without_extension)
                    print('Recording filename added to list of recordings filenames to retry to delete.')

                # Remove thumbnail files from disk
                for thumbnail_format in thumbnail_formats:
                    try:
                        thumbnail_filename = f'{recording_filename_without_extension}{thumbnail_format}{thumbnail_file_extension}'
                        thumbnail_filepath = os.path.join(recordings_folder_path, 'thumbnails', thumbnail_filename)
                        thumbnail_size_mebibytes = bytes_to_mebibytes(os.path.getsize(thumbnail_filepath))
                        os.remove(thumbnail_filepath)

                        # Update free space value
                        current_free_space_mebibytes += thumbnail_size_mebibytes
                    except OSError as e:
                        print('Exception caught while trying to delete thumbnail. Exception:', e)

            except Exception as e:
                print('Exception caught while freeing space. Exception:', e)


    def stop(self):
        configManager.remove_observer(self)
        self._http_session.close()
        self._running = False
