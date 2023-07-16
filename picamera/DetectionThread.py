from threading import Thread
import os
import time
import cv2
from picamera.motionDetection import MotionDetector
from picamera.objectDetection import ObjectDetector
from picamera.detection_typing import BoundingBox

# DEBUG VARIABLES
debug_display_video_window = False
debug_draw_detection_boxes_on_video = False

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

class DetectionThread(Thread):
    def __init__(self, picam):
        Thread.__init__(self)
        self._running = True
        self._picam = picam

        self._objects_to_detect_with_weights = {
            'car': 5,
            'motorcycle': 4,
            'bicycle': 3,
            'person': 2,
            'cat': 1,
            'dog': 1
        }

        self._motion_detection_areas: tuple[BoundingBox, ...] = (
            # (800, 50, 1120, 200),
            # (800, 350, 1120, 300),
            #
            # (250, 0, 200, 200),
            # (450, 0, 725, 1080),
            # (0, 500, 450, 1080),
            # (1175, 300, 350, 1080),
            # (1525, 350, 250, 1080),
            # (1775, 400, 250, 300),
        )

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
            objects_to_detect=list(self._objects_to_detect_with_weights.keys()),
            min_frames_for_detection=5
        )

    def run(self):
        if debug_display_video_window:
            cv2.startWindowThread()
            cv2.namedWindow("debug", cv2.WINDOW_NORMAL)

        self._motion_detector.reset_detector()
        self._object_detector.reset_detector()

        RECORDING_PHASE_DURATION_IN_SECONDS = 8
        MAX_RECORDING_PHASE_DURATION_IN_SECONDS = 120
        CHECKING_PHASE_DURATION_IN_SECONDS = 60

        previous_object_type_detected = None
        time_when_object_was_detected = time.time()
        time_when_recording_phase_started = time.time()
        time_when_checking_phase_started = time.time()
        is_in_recording_phase = False  # Recording video of detection
        is_in_checking_phase = False  # Checking if the last object detected is still there

        # Grab the array representing the frame
        while self._running and (frame := self._picam.get_frame()) is not None:
            if is_in_recording_phase:
                current_time = time.time()
                seconds_elapsed_since_object_was_detected = current_time - time_when_object_was_detected
                seconds_elapsed_since_recording_phase_started = current_time - time_when_recording_phase_started
                if (
                        seconds_elapsed_since_object_was_detected >= RECORDING_PHASE_DURATION_IN_SECONDS
                        or seconds_elapsed_since_recording_phase_started >= MAX_RECORDING_PHASE_DURATION_IN_SECONDS
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

                if seconds_elapsed_since_checking_phase_started >= CHECKING_PHASE_DURATION_IN_SECONDS:
                    is_in_checking_phase = False
                    print('-------------- leaving checking phase')
                    previous_object_type_detected = None

            motion_detection = self._motion_detector.detect(frame=frame, detection_areas=self._motion_detection_areas)

            if debug_draw_detection_boxes_on_video:
                for motion_detection_area in self._motion_detection_areas:
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
                                                                       self._objects_to_detect_with_weights)

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
                self._picam.start_recording()
                print('start recording')
                # SEND NOTIFICATION OF DETECTION
                print('SENDING NOTIFICATION OF DETECTION FOR OJBECT', object_to_notify_type)

            elif is_in_checking_phase and object_to_notify_type == previous_object_type_detected:
                time_when_checking_phase_started = time.time()

            time_when_object_was_detected = time.time()
            previous_object_type_detected = object_to_notify_type

            if debug_display_video_window:
                cv2.imshow('debug', frame)

    def stop(self):
        self._running = False