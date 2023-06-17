import cv2
from typing import Optional, Union, Any
from detection_typing import Frame, BoundingBox, ObjectDetectionResult

class ObjectDetector:
    def __init__(self) -> None:
        self._dataset_class_names: list[str] = []
        self._net = None
        self._nb_frames_for_detected_objects: dict[str, int] = {}
        self._configuration: dict[str, Any] = {
            'detection_confidence_threshold': 0.65,
            'detection_non_maximum_suppression_threshold': 0.1,
            'objects_to_detect': [],
            'min_frames_for_detection': 1,
        }


    def configure_dataset(self, dataset_file_path: str) -> None:
        with open(dataset_file_path, 'rt') as file:
            self._dataset_class_names = file.read().splitlines()
            self._configuration['objects_to_detect'] = self._dataset_class_names.copy()

    def configure_neural_network(
        self,
        neural_network_weights_file_path: str,
        neural_network_config_file_path: str,
        input_size: tuple,
        input_scale: float,
        input_mean: tuple,
        input_swap_rb: bool
    ) -> None:
        self._net = cv2.dnn_DetectionModel(neural_network_weights_file_path, neural_network_config_file_path)
        self._net.setInputSize(input_size)
        self._net.setInputScale(input_scale)
        self._net.setInputMean(input_mean)
        self._net.setInputSwapRB(input_swap_rb)

    def configure_detection(
        self,
        confidence_threshold: float = None,
        non_maximum_suppression_threshold: float = None,
        objects_to_detect: list[str] = None,
        min_frames_for_detection: int = None
    ) -> None:
        if confidence_threshold is not None:
            if confidence_threshold < 0 or confidence_threshold > 1:
                raise ValueError('The confidence threshold is a percentage and must be between 0 and 1')
            self._configuration['detection_confidence_threshold'] = confidence_threshold

        if non_maximum_suppression_threshold is not None:
            if non_maximum_suppression_threshold < 0 or non_maximum_suppression_threshold > 1:
                raise ValueError('The non maximum suppression threshold is a percentage and must be between 0 and 1')
            self._configuration['detection_non_maximum_suppression_threshold'] = non_maximum_suppression_threshold

        if objects_to_detect is not None:
            self._configuration['objects_to_detect'] = objects_to_detect

        if min_frames_for_detection is not None:
            if min_frames_for_detection < 1:
                raise ValueError('The minimum number of frames for the object to be considered detected cannot be less than 1')
            self._configuration['min_frames_for_detection'] = min_frames_for_detection

    def detect(self, frame: Optional[Frame], interest_areas: Union[list[BoundingBox], tuple[BoundingBox, ...]] = tuple()) -> list[ObjectDetectionResult]:
        if len(self._dataset_class_names) == 0:
            raise ValueError('A dataset must be provided before being able to detect objects.')

        if self._net is None:
            raise ValueError('A neural network must be provided before being able to detect objects.')

        # if we pass no frame, immediately return empty list
        if frame is None or len(frame) == 0:
            return []

        interest_areas = ((0, 0, len(frame[0]), len(frame)),) if len(interest_areas) == 0 else interest_areas

        # if we are not interested in detecting any type of object, we can shortcut the process and return immediately
        if len(self._configuration['objects_to_detect']) == 0:
            return [[] for i in range(len(interest_areas))]

        distinct_detected_objects_among_all_sub_frames: set[str] = set()
        all_detections: list[ObjectDetectionResult] = []

        for interest_area_x, interest_area_y, interest_area_width, interest_area_height in interest_areas:
            sub_frame = frame[interest_area_y:interest_area_y + interest_area_height, interest_area_x:interest_area_x + interest_area_width]

            class_ids, confidences, bounding_boxes = self._net.detect(
                frame = sub_frame,
                confThreshold = self._configuration['detection_confidence_threshold'],
                nmsThreshold = self._configuration['detection_non_maximum_suppression_threshold']
            )

            if len(class_ids) == 0:
                all_detections.append([])
                continue

            detection_results_for_sub_frame = []

            for class_id, confidence, bounding_box_relative_to_sub_frame in zip(class_ids.flatten(), confidences.flatten(), bounding_boxes):
                class_name: str = self._dataset_class_names[class_id - 1]

                if class_name in self._configuration['objects_to_detect']:
                    bounding_box_x, bounding_box_y, bounding_box_width, bounding_box_height = bounding_box_relative_to_sub_frame
                    bounding_box_relative_to_frame: BoundingBox = (interest_area_x + bounding_box_x, interest_area_y + bounding_box_y, bounding_box_width, bounding_box_height)

                    detection_results_for_sub_frame.append((class_name, confidence, bounding_box_relative_to_frame))

                    distinct_detected_objects_among_all_sub_frames.add(class_name)

            if len(detection_results_for_sub_frame) == 0:
                all_detections.append([])
                continue

            all_detections.append(detection_results_for_sub_frame)

        # If the object already exists in the dictionary, add 1 to the number of frames during which the object was detected,
        # else add the object to the dictionary with a value of 1
        self._nb_frames_for_detected_objects = {object_type: self._nb_frames_for_detected_objects.get(object_type, 0) + 1 for object_type in distinct_detected_objects_among_all_sub_frames}

        # Check if an object was detected for a minimum number of frames

        detected_objects_with_min_frames: tuple[str, ...] = tuple(object_type for object_type in self._nb_frames_for_detected_objects if self._nb_frames_for_detected_objects[object_type] >= self._configuration['min_frames_for_detection'])

        print(self._nb_frames_for_detected_objects)

        if len(detected_objects_with_min_frames) == 0:
            return [[] for i in range(len(interest_areas))]

        detections_with_min_frames: list[ObjectDetectionResult] = []

        def take_detections_with_min_frames(detection_of_one_object: tuple[str, float, BoundingBox]) -> bool:
            object_type, _, _ = detection_of_one_object
            return object_type in detected_objects_with_min_frames

        for detections_in_sub_frame in all_detections:
            filtered_detections: ObjectDetectionResult = list(filter(take_detections_with_min_frames, detections_in_sub_frame))
            detections_with_min_frames.append(filtered_detections)

        return detections_with_min_frames

    def resetDetector(self) -> None:
        self._nb_frames_for_detected_objects = {}
