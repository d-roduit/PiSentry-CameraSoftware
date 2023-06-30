import cv2
import numpy as np
from typing import Optional, Any, Iterator, Generator, Union
from picamera.detection_typing import BoundingBox, Frame, Point

class MotionDetector:
    def __init__(self) -> None:
        self._configuration: dict[str, Any] = {
            'delta_threshold': 9,
            'min_area_for_motion': 1000,
            'min_frames_for_motion': 20,
        }
        self.average_frame: Optional[Frame] = None
        self._nb_consecutive_frames_with_motion_detected: int = 0

    def configure_detection(
        self,
        delta_threshold: float = None,
        min_area_for_motion: float = None,
        min_frames_for_motion: int = None,
    ) -> None:
        if delta_threshold is not None:
            if delta_threshold < 0:
                raise ValueError('The delta threshold cannot be negative')
            self._configuration['delta_threshold'] = delta_threshold

        if min_area_for_motion is not None:
            if min_area_for_motion < 0:
                raise ValueError('The minimum area for motion cannot be negative')
            self._configuration['min_area_for_motion'] = min_area_for_motion

        if min_frames_for_motion is not None:
            if min_frames_for_motion < 1:
                raise ValueError('The minimum number of frames for motion to be considered detected cannot be less than 1')
            self._configuration['min_frames_for_motion'] = min_frames_for_motion


    def detect(self, frame: Frame, detection_areas: Union[list[BoundingBox], tuple[BoundingBox, ...]] = tuple()) -> tuple:
        # if we pass no frame, immediately return
        if frame is None:
            print('returning cause frame is None')
            self._nb_consecutive_frames_with_motion_detected = 0
            return tuple()

        # convert the frame to grayscale and blur it
        gray_frame: Frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blured_frame: Frame = cv2.GaussianBlur(src=gray_frame, ksize=(9, 9), sigmaX=0)

        # if the average frame is None, initialize it
        if self.average_frame is None:
            self.average_frame = blured_frame.copy().astype("float")
            self._nb_consecutive_frames_with_motion_detected = 0
            print('returning cause average_frame is None')
            return tuple()

        # accumulate the weighted average between the current frame and previous frames,
        # then compute the difference between the current frame and running average
        cv2.accumulateWeighted(src=blured_frame, dst=self.average_frame, alpha=0.5)
        frame_delta: Frame = cv2.absdiff(blured_frame, cv2.convertScaleAbs(self.average_frame))

        # threshold the delta image with binary threshold, meaning making pixels either black or white
        # black pixels represent what didn't move, white pixels what did move
        _, thresholded_frame = cv2.threshold(src=frame_delta, thresh=self._configuration["delta_threshold"], maxval=255, type=cv2.THRESH_BINARY)

        # dilate the thresholded image to fill in holes
        dilated_frame: Frame = cv2.dilate(src=thresholded_frame, kernel=np.ones((5, 5)), iterations=2)

        # in OpenCV 2, findContours(...) returns two parameters. But if we were to use OpenCV >= 3.2,
        # findContours(...) would return three parameters and the contours value would be
        # the second return parameters (meaning index 1 : [1])
        movement_contours, _ = cv2.findContours(image=dilated_frame.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        # no contours were detected
        if len(movement_contours) == 0:
            print('------------------------- AUCUN MOUVEMENT CONTOURS -------------------------')
            self._nb_consecutive_frames_with_motion_detected = 0
            print('returning cause len(movement_contours) == 0')
            return tuple()

        # 1. is there movement in detection areas ? -> look if the center point of the bounding box is in any detection areas

        movement_bounding_boxes: list[BoundingBox] = list(map(lambda contour: cv2.boundingRect(contour), movement_contours))

        movement_bounding_boxes_in_detection_area: list[BoundingBox] = []
        movement_contours_in_detection_area: list[np.ndarray] = []

        detection_areas = ((0, 0, len(frame[0]), len(frame)),) if len(detection_areas) == 0 else detection_areas

        for movement_bounding_box, movement_contour in zip(movement_bounding_boxes, movement_contours):
            for detection_area in detection_areas:
                if self._bounding_box_contains_point(
                    bounding_box = detection_area,
                    point = self._bounding_box_center(movement_bounding_box)
                ):
                    movement_bounding_boxes_in_detection_area.append(movement_bounding_box)
                    movement_contours_in_detection_area.append(movement_contour)
                    break

        is_motion_detected_in_detection_area: bool = len(movement_bounding_boxes_in_detection_area) > 0

        if not is_motion_detected_in_detection_area:
            self._nb_consecutive_frames_with_motion_detected = 0
            print('returning cause is_motion_detected_in_detection_area is False')
            return tuple()

        if len(movement_bounding_boxes_in_detection_area) == 0:
            print('movement_bounding_boxes_in_detection_area LEN IS 0 ------------------------ /////////////////////////////')

        # 2. is movement big enough (e.g. we don't want to capture tiny flowers moving) ?

        bounding_boxes_and_contours_iterator: Iterator = zip(movement_bounding_boxes_in_detection_area, movement_contours_in_detection_area)
        motions_with_min_area_generator_filter: Generator = ((bounding_box, contour) for bounding_box, contour in bounding_boxes_and_contours_iterator if cv2.contourArea(contour) >= self._configuration["min_area_for_motion"])
        motions_with_min_area: list[tuple] = list(zip(*motions_with_min_area_generator_filter))

        if len(motions_with_min_area) == 0:
            print('returning cause len(motions_with_min_area) == 0')
            self._nb_consecutive_frames_with_motion_detected = 0
            return tuple()

        # 3. is this a movement which lasts (for several consecutive frames) ?  -> if so, there is real movement !

        self._nb_consecutive_frames_with_motion_detected += 1

        if self._nb_consecutive_frames_with_motion_detected < self._configuration["min_frames_for_motion"]:
            print('returning cause nb_consecutive_frames_with_motion_detected < self._configuration["min_frames_for_motion"]')
            return tuple()

        print(f'motion was detected in {self._nb_consecutive_frames_with_motion_detected} consecutive frames')

        # Remove the bounding boxes which are too small compared to the biggest bounding_box

        bounding_boxes_with_min_area, contours_with_min_area = motions_with_min_area

        bounding_boxes_sorted_by_area_desc, contours_sorted_by_area_desc = zip(*sorted(zip(bounding_boxes_with_min_area, contours_with_min_area), key=lambda zipped_item: self._bounding_box_area(zipped_item[0]), reverse=True))

        area_of_biggest_bounding_box = self._bounding_box_area(bounding_boxes_sorted_by_area_desc[0])

        big_enough_boxes, big_enough_contours = zip(*((bounding_box, contour) for bounding_box, contour in zip(bounding_boxes_sorted_by_area_desc, contours_sorted_by_area_desc) if self._bounding_box_area(bounding_box) >= area_of_biggest_bounding_box / 4))

        return big_enough_boxes, big_enough_contours

    def reset_detector(self) -> None:
        self.average_frame = None
        self._nb_consecutive_frames_with_motion_detected = 0


    def _bounding_box_area(self, bounding_box: BoundingBox) -> int:
        if len(bounding_box) != 4:
            raise ValueError('A bounding box must have four values (x, y, width, height)')

        x, y, width, height = bounding_box
        return height * width


    def _bounding_box_contains_point(self, bounding_box: BoundingBox, point: Point) -> bool:
        if len(bounding_box) != 4:
            raise ValueError('A bounding box must have four values (x, y, width, height)')

        if len(point) != 2:
            raise ValueError('A point must have two coordinates (x, y)')

        bounding_box_x, bounding_box_y, bounding_box_width, bounding_box_height = bounding_box
        point_x, point_y = point

        return (
            point_x in range(bounding_box_x, bounding_box_x + bounding_box_width + 1) # + 1 is necessary as range() is exclusive at the end of the range
            and point_y in range(bounding_box_y, bounding_box_y + bounding_box_height + 1) # + 1 is necessary as range() is exclusive at the end of the range
        )

    def _bounding_box_center(self, bounding_box: BoundingBox) -> Point:
        if len(bounding_box) != 4:
            raise ValueError('A bounding box must have four values (x, y, width, height)')

        x, y, width, height = bounding_box
        center_x = x + (width // 2)
        center_y = y + (height // 2)
        return center_x, center_y
