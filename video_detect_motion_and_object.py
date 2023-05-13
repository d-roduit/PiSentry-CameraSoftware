import os
import cv2
import numpy as np
from motionDetection import MotionDetector
from objectDetection import ObjectDetector
# import time

if __name__ == "__main__":
    cv2.startWindowThread()
    cap = cv2.VideoCapture('videos/video10.MOV')
    motionDetector = MotionDetector(delta_threshold = 9, min_area_for_motion = 1000, min_frames_for_motion = 20)
    objectDetector = ObjectDetector()

    directory_path = os.path.dirname(__file__)
    coco_dataset_file_path = os.path.join(directory_path, 'objectDetection', 'coco.names')
    neural_network_weights_file_path = os.path.join(directory_path, 'objectDetection', 'frozen_inference_graph.pb')
    neural_network_config_file_path = os.path.join(directory_path, 'objectDetection', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')

    objectDetector.configure_dataset(coco_dataset_file_path)
    objectDetector.configure_neural_network(
        neural_network_weights_file_path = neural_network_weights_file_path,
        neural_network_config_file_path = neural_network_config_file_path,
        input_size = (320, 320),
        input_scale = 1.0 / 127.5,
        input_mean = (127.5, 127.5, 127.5),
        input_swap_rb = True
    )
    objectDetector.configure_detection(confidence_threshold = 0.65, non_maximum_suppression_threshold = 0.1, objects_to_detect=['person','bicycle','car','motorcycle','cat','dog'])

    height_in_pixel = 480
    width_in_pixel = 640
    nb_colors_per_pixel = 3
    frame_array_config = (height_in_pixel, width_in_pixel, nb_colors_per_pixel)
    black_frame = np.zeros(frame_array_config)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame_width, frame_height))

    # nb_frames = 0
    # total_time = 0

    while cap.isOpened():
        # grab the array representing the image
        ret, frame = cap.read()

        # start = time.process_time()
        motion_detection = motionDetector.detect(frame)
        # total_time += (time.process_time() - start)
        # nb_frames += 1

        cv2.rectangle(img=frame, pt1=(0,0), pt2=(1920, 1080), color=(0, 0, 0), thickness=2)

        if motion_detection is None:
            print('motion detection is None _________________________')

        if motion_detection is not None:
            movement_bounding_boxes, _ = motion_detection

            print('len(movement_bounding_boxes):', len(movement_bounding_boxes))

            for movement_bounding_box in movement_bounding_boxes:
                movement_x, movement_y, movement_width, movement_height = movement_bounding_box

                # if movement_y + movement_height / 2 not in range(500, 1920):
                #     continue

                # draw the bounding box on the frame
                movement_top_left_corner = (movement_x, movement_y)
                movement_bottom_right_corner = (movement_x + movement_width, movement_y + movement_height)
                cv2.rectangle(img=frame, pt1=movement_top_left_corner, pt2=movement_bottom_right_corner, color=(0, 255, 0), thickness=2)


                # 4. make object detection on all big enough movements


                object_detection = objectDetector.detect(frame[movement_top_left_corner[1]:movement_bottom_right_corner[1], movement_top_left_corner[0]:movement_bottom_right_corner[0]])

                if object_detection is not None:
                    object_types, object_confidences, object_bounding_boxes = object_detection

                    print('len(object_types):', len(object_types))
                    sorted_object_confidences, sorted_object_types = zip(*sorted(zip(object_confidences, object_types)))
                    for obj_conf, obj_type in zip(sorted_object_confidences, sorted_object_types):
                        print('obj_type:', obj_type, ' -> ', obj_conf)

                    # print('object_types:', object_types)
                    # print('object_confidences:', object_confidences)
                    # print('object_bounding_boxes:', object_bounding_boxes)
                    # print('\n')

                    # 5. if objects were detected, take only the detection with the highest confidence

                    for object_type, object_confidence, object_bounding_box in zip(object_types, object_confidences, object_bounding_boxes):
                        object_x, object_y, object_width, object_height = object_bounding_box

                        # if object_confidence < 0.65:
                        #     continue

                        # draw the bounding box on the frame
                        object_top_left_corner = (movement_x + object_x, movement_y + object_y)
                        object_bottom_right_corner = (movement_x + object_x + object_width, movement_y + object_y + object_height)
                        cv2.rectangle(img=frame, pt1=object_top_left_corner, pt2=object_bottom_right_corner, color=(0, 125, 0), thickness=2)
                        cv2.putText(frame, object_type.upper(), (object_top_left_corner[0] + 10, object_top_left_corner[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 125, 0), 2)
                        cv2.putText(frame, str(round(object_confidence * 100, 2)), (object_top_left_corner[0] + 200, object_top_left_corner[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 125, 0), 2)


        # out.write(frame)

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.imshow('output', frame)

        # print('moyenne:')
        # print(total_time / nb_frames)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
