import datetime
import cv2
import os
from picamera.detection_typing import BoundingBox

def time_in_range(time_to_check, start_time, end_time) -> bool:
    """Return true if time_to_check is in the range [start_time, end_time]"""
    if not isinstance(time_to_check, datetime.time) or not isinstance(start_time, datetime.time) or not isinstance(end_time, datetime.time):
        raise ValueError('All function parameters must be of type `datetime.time`')

    if start_time == end_time:
        return True
    elif start_time < end_time:
        return start_time <= time_to_check < end_time
    else:
        return start_time <= time_to_check or time_to_check < end_time

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

def write_frame_to_file(frame, folderpath, filename, cv2_params = None):
    if not os.path.isabs(folderpath):
        raise ValueError('The folder path must be an absolute path. Received:', folderpath)

    os.makedirs(name=folderpath, exist_ok=True)

    filepath = os.path.join(folderpath, filename)

    cv2.imwrite(filepath, frame, cv2_params)


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