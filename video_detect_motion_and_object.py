import os
import cv2
from motionDetection import MotionDetector
from objectDetection import ObjectDetector
import time
from detection_typing import BoundingBox

if __name__ == "__main__":
    cv2.startWindowThread()
    cap = cv2.VideoCapture('videos/pi_record_1920x1080_7.mp4')

    directory_path = os.path.dirname(__file__)
    coco_dataset_file_path = os.path.join(directory_path, 'objectDetection', 'coco.names')
    neural_network_weights_file_path = os.path.join(directory_path, 'objectDetection', 'frozen_inference_graph.pb')
    neural_network_config_file_path = os.path.join(directory_path, 'objectDetection', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')

    objects_to_detect_with_weights = {
        'car': 5,
        'motorcycle': 4,
        'bicycle': 3,
        'person': 2,
        'cat': 1,
        'dog': 1
    }

    motion_detection_areas: tuple[BoundingBox, ...] = (
        (800, 50, 1120, 200),
        (800, 350, 1120, 300),
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

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    frame_number = 0

    total_time = 0
    total_detection_time = 0
    number_frames_with_detection = 1

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


    while cap.isOpened():
        frame_number += 1
        print('\n---------------------------------')
        print('Frame no:\t', frame_number)
        print('---------------------------------\n')
        print('--------- Motion ---------\n')

        # grab the array representing the image
        ret, frame = cap.read()

        start_time = time.process_time()

        motion_detection = motionDetector.detect(frame = frame, detection_areas = motion_detection_areas)

        for motion_detection_area in motion_detection_areas:
            print('motion_detection_area:', motion_detection_area)
            draw_on_frame(frame, motion_detection_area, (200, 100, 0))

        if len(motion_detection) == 0:
            print('no motion detection !!')

        frame_has_detection = False

        if len(motion_detection) > 0:
            movement_bounding_boxes, _ = motion_detection

            print('number of movement bounding boxes:', len(movement_bounding_boxes))

            frame_has_detection = True

            start_detection = time.process_time()
            object_detection: list[list[tuple]] = objectDetector.detect(frame = frame, interest_areas = movement_bounding_boxes)
            total_detection_time += time.process_time() - start_detection

            for movement_bounding_box in movement_bounding_boxes:
                draw_on_frame(frame, movement_bounding_box)


            if len(object_detection) > 0:
                for object_detections_in_sub_frame in object_detection:
                    print('\n--------- Object ---------\n')

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


        if frame_has_detection:
            total_time += time.process_time() - start_time
            number_frames_with_detection += 1

        print('avg detection time for frame:\t', total_detection_time / number_frames_with_detection)
        print('avg total time for frame:\t', total_time / number_frames_with_detection)
        print('number_frames_with_detection:', number_frames_with_detection)

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.imshow('output', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
