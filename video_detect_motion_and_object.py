import os
import cv2
from picamera.motionDetection import MotionDetector
from picamera.objectDetection import ObjectDetector
import time
from picamera.detection_typing import BoundingBox
from picamera import picam

if __name__ == "__main__":
    cv2.startWindowThread()

    directory_path = os.path.dirname(__file__)
    object_detection_directory = os.path.join(directory_path, 'picamera', 'objectDetection')
    coco_dataset_file_path = os.path.join(object_detection_directory, 'coco.names')
    neural_network_weights_file_path = os.path.join(object_detection_directory, 'frozen_inference_graph.pb')
    neural_network_config_file_path = os.path.join(object_detection_directory, 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')

    objects_to_detect_with_weights = {
        'car': 5,
        'motorcycle': 4,
        'bicycle': 3,
        'person': 2,
        'cat': 1,
        'dog': 1
    }

    motion_detection_areas: tuple[BoundingBox, ...] = (
        # (800, 50, 1120, 200),
        # (800, 350, 1120, 300),
    )

    motionDetector = MotionDetector()
    objectDetector = ObjectDetector()

    motionDetector.configure_detection(
        delta_threshold = 9,
        min_area_for_motion = 1000,
        min_frames_for_motion = 20
    )

    objectDetector.configure_dataset(coco_dataset_file_path)
    objectDetector.configure_neural_network(
        neural_network_weights_file_path = neural_network_weights_file_path,
        neural_network_config_file_path = neural_network_config_file_path,
        input_size = (320, 320),
        input_scale = 1.0 / 127.5,
        input_mean = (127.5, 127.5, 127.5),
        input_swap_rb = True
    )
    objectDetector.configure_detection(
        confidence_threshold = 0.65,
        non_maximum_suppression_threshold = 0.1,
        objects_to_detect = list(objects_to_detect_with_weights.keys()),
        min_frames_for_detection = 10
    )

    frame_number = 0

    def draw_on_frame(frame, bounding_box, bounding_box_color = (0, 255, 0), object_type = None, confidence = None, text_color = (0, 125, 0)):
        x, y, width, height = bounding_box
        top_left_corner = (x, y)
        bottom_right_corner = (x + width, y + height)
        cv2.rectangle(img=frame, pt1=top_left_corner, pt2=bottom_right_corner, color=bounding_box_color, thickness=2)

        if object_type is not None:
            cv2.putText(frame, object_type.upper(), (top_left_corner[0] + 10, top_left_corner[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, text_color, 2)

        if confidence is not None:
            cv2.putText(frame, str(round(confidence * 100, 2)), (top_left_corner[0] + 200, top_left_corner[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, text_color, 2)


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

    TIME_PERIOD_TO_CHECK_IF_SAME_OBJECT_IS_STILL_THERE_IN_SECONDS = 10
    WAITING_TIME_BEFORE_RESUMING_DETECTION_IN_SECONDS = 10
    previous_object_type_detected = None
    time_when_object_was_detected = time.time()
    time_when_checking_phase_started = time.time()
    is_in_waiting_phase = False # Waiting before resuming detection
    is_in_checking_phase = False # Checking if the last object detected is still there
    can_detect = True

    # grab the array representing the image
    while (frame := picam.get_frame()) is not None:
        frame_number += 1
        # print('\n---------------------------------')
        # print('Frame no:\t', frame_number)
        # print('---------------------------------\n')
        # print('--------- Motion ---------\n')

        if is_in_waiting_phase:
            can_detect = False
            current_time = time.time()
            seconds_elapsed_since_detection = current_time - time_when_object_was_detected

            if seconds_elapsed_since_detection >= WAITING_TIME_BEFORE_RESUMING_DETECTION_IN_SECONDS:
                is_in_waiting_phase = False
                is_in_checking_phase = True
                print('-------------- leaving waiting phase')
                print('-------------- entering checking phase')
                time_when_checking_phase_started = time.time()

        elif is_in_checking_phase:
            can_detect = True
            current_time = time.time()
            seconds_elapsed_since_checking_phase_started = current_time - time_when_checking_phase_started

            if seconds_elapsed_since_checking_phase_started >= TIME_PERIOD_TO_CHECK_IF_SAME_OBJECT_IS_STILL_THERE_IN_SECONDS:
                is_in_checking_phase = False
                print('-------------- leaving checking phase')
                previous_object_type_detected = None


        if can_detect:
            motion_detection = motionDetector.detect(frame = frame, detection_areas = motion_detection_areas)

            for motion_detection_area in motion_detection_areas:
                draw_on_frame(frame, motion_detection_area, (200, 100, 0))

            if len(motion_detection) == 0:
                pass
                # print('no motion detection !!')

            if len(motion_detection) > 0:
                movement_bounding_boxes, _ = motion_detection

                print('number of movement bounding boxes:', len(movement_bounding_boxes))

                print('\n--------- Object ---------\n')

                object_detection: list[list[tuple]] = objectDetector.detect(frame = frame, interest_areas = movement_bounding_boxes)

                for movement_bounding_box in movement_bounding_boxes:
                    draw_on_frame(frame, movement_bounding_box)

                if len(object_detection) > 0:
                    for object_detections_in_sub_frame in object_detection:
                        print('\n--------- Object detection in sub frame\n')

                        print('nb objects detected in sub frame:', len(object_detections_in_sub_frame))

                        if len(object_detections_in_sub_frame) == 0:
                            continue

                        for detection_result_of_one_object in object_detections_in_sub_frame:
                            object_type, object_confidence, object_bounding_box = detection_result_of_one_object
                            draw_on_frame(frame, object_bounding_box, (0, 125, 0), object_type, object_confidence)

                    object_to_notify = get_object_with_highest_weight_and_area(object_detection, objects_to_detect_with_weights)

                    print('object_with_highest_weight_and_area:', object_to_notify)

                    if len(object_to_notify) > 0:
                        object_to_notify_type, object_to_notify_confidence, object_to_notify_bounding_box = object_to_notify
                        draw_on_frame(frame, object_to_notify_bounding_box, (0, 100, 200), object_to_notify_type, object_to_notify_confidence, (0, 100, 200))

                        print(object_to_notify_type, 'DETECTED')
                        print('previous_object_type_detected:', previous_object_type_detected)
                        print('is_in_checking_phase:', is_in_checking_phase)

                        must_notify_detection = (
                            not is_in_checking_phase
                            or (is_in_checking_phase and object_to_notify_type != previous_object_type_detected)
                        )

                        if must_notify_detection:
                            # SEND NOTIFICATION OF DETECTION
                            print('SENDING NOTIFICATION OF DETECTION FOR OJBECT', object_to_notify_type)

                        previous_object_type_detected = object_to_notify_type
                        is_in_checking_phase = False
                        is_in_waiting_phase = True
                        print('-------------- entering waiting phase')
                        time_when_object_was_detected = time.time()
                        motionDetector.resetDetector()
                        objectDetector.resetDetector()


        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.imshow('output', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Closes all the frames
    cv2.destroyAllWindows()
