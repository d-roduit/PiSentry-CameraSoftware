import cv2
import numpy as np
from typing import Optional, Any, Iterator, Generator

class MotionDetector:
    def __init__(self, delta_threshold: float = 9, min_area_for_motion: float = 1000, min_frames_for_motion: int = 20) -> None:
        cv2.startWindowThread()
        self.configuration: dict[str, Any] = {
            'delta_threshold': delta_threshold,
            'min_area_for_motion': min_area_for_motion,
            'min_frames_for_motion': min_frames_for_motion,
        }
        self.average_frame: Optional[np.ndarray] = None
        self.nb_consecutive_frames_with_motion_detected: int = 0

    def detect(self, frame: np.ndarray) -> Optional[tuple]:
        # if we pass no frame, immediately return None
        if frame is None:
            print('returning cause frame is None')
            self.nb_consecutive_frames_with_motion_detected = 0
            return None

        # convert the frame to grayscale and blur it
        gray_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blured_frame: np.ndarray = cv2.GaussianBlur(src=gray_frame, ksize=(9, 9), sigmaX=0)

        # if the average frame is None, initialize it
        if self.average_frame is None:
            self.average_frame = blured_frame.copy().astype("float")
            self.nb_consecutive_frames_with_motion_detected = 0
            print('returning cause average_frame is None')
            return None

        # accumulate the weighted average between the current frame and previous frames,
        # then compute the difference between the current frame and running average
        cv2.accumulateWeighted(src=blured_frame, dst=self.average_frame, alpha=0.5)
        frame_delta: np.ndarray = cv2.absdiff(blured_frame, cv2.convertScaleAbs(self.average_frame))

        # threshold the delta image with binary threshold, meaning making pixels either black or white
        # black pixels represent what didn't move, white pixels what did move
        _, thresholded_frame = cv2.threshold(src=frame_delta, thresh=self.configuration["delta_threshold"], maxval=255, type=cv2.THRESH_BINARY)

        # dilate the thresholded image to fill in holes
        dilated_frame: np.ndarray = cv2.dilate(src=thresholded_frame, kernel=np.ones((5, 5)), iterations=2)


        cv2.namedWindow("output2", cv2.WINDOW_NORMAL)
        cv2.imshow('output2', dilated_frame)

        # print('moyenne:')
        # print(total_time / nb_frames)

        # Press Q on keyboard to  exit
        cv2.waitKey(25)


        # in OpenCV 2, findContours(...) returns two parameters. But if we were to use OpenCV >= 3.2,
        # findContours(...) would return three parameters and the contours value would be
        # the second return parameters (meaning index 1 : [1])
        movement_contours, _ = cv2.findContours(image=dilated_frame.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        # no contours were detected
        if len(movement_contours) == 0:
            print('------------------------- AUCUN MOUVEMENT CONTOURS -------------------------')
            self.nb_consecutive_frames_with_motion_detected = 0
            print('returning cause len(movement_contours) == 0')
            return None

        # 1. is there movement in detection area ? -> look if the center point of the bounding box is in the detection area

        movement_bounding_boxes: list[tuple[int, int, int, int]] = list(map(lambda contour: cv2.boundingRect(contour), movement_contours))

        motions_in_detection_area: list[tuple] = list(zip(*((bounding_box, contour) for bounding_box, contour in zip(movement_bounding_boxes, movement_contours) if bounding_box[1] + bounding_box[3] // 2 in range(0, 1080))))

        is_motion_detected_in_detection_area: bool = len(motions_in_detection_area) > 0

        if not is_motion_detected_in_detection_area:
            self.nb_consecutive_frames_with_motion_detected = 0
            print('returning cause is_motion_detected_in_detection_area is False')
            return None

        movement_bounding_boxes_in_detection_area, movement_contours_in_detection_area = motions_in_detection_area

        if len(movement_bounding_boxes_in_detection_area) == 0:
            print('movement_bounding_boxes_in_detection_area LEN IS 0 ------------------------ /////////////////////////////')

        # 2. is movement big enough (e.g. we don't want to capture tiny flowers moving) ?

        bounding_boxes_and_contours_iterator: Iterator = zip(movement_bounding_boxes_in_detection_area, movement_contours_in_detection_area)
        big_enough_motions_generator_filter: Generator = ((bounding_box, contour) for bounding_box, contour in bounding_boxes_and_contours_iterator if cv2.contourArea(contour) >= self.configuration["min_area_for_motion"])
        big_enough_motions: list[tuple] = list(zip(*big_enough_motions_generator_filter))

        if len(big_enough_motions) == 0:
            print('returning cause len(big_enough_motions) == 0')
            self.nb_consecutive_frames_with_motion_detected = 0
            return None

        # 3. is this a movement which lasts (for several consecutive frames) ?  -> if so, there is real movement !

        self.nb_consecutive_frames_with_motion_detected += 1

        if self.nb_consecutive_frames_with_motion_detected < self.configuration["min_frames_for_motion"]:
            print('returning cause nb_consecutive_frames_with_motion_detected < self.configuration["min_frames_for_motion"]')
            return None

        print(f'motion was detected in {self.nb_consecutive_frames_with_motion_detected} consecutive frames')


        big_enough_bounding_boxes, big_enough_contours = big_enough_motions

        return big_enough_bounding_boxes, big_enough_contours
