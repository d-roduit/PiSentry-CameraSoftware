import cv2
from typing import Optional

DetectionResult = Optional[tuple[list[str], list[float], list[tuple[int, int, int, int]]]]

class ObjectDetector:
    def __init__(self) -> None:
        self._dataset_class_names: list[str] = []
        self._net = None
        self._detection_confidence_threshold: float = 0.65
        self._detection_non_maximum_suppression_threshold: float = 0.1
        self._objects_to_detect: list[str] = []

    def configure_dataset(self, dataset_file_path: str) -> None:
        with open(dataset_file_path, 'rt') as file:
            self._dataset_class_names = file.read().splitlines()
            self._objects_to_detect = self._dataset_class_names.copy()

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
            objects_to_detect: list[str] = None
    ) -> None:
        if confidence_threshold is not None:
            self._detection_confidence_threshold = confidence_threshold

        if non_maximum_suppression_threshold is not None:
            self._detection_non_maximum_suppression_threshold = non_maximum_suppression_threshold

        if objects_to_detect is not None:
            self._objects_to_detect = objects_to_detect

    def detect(self, frame) -> DetectionResult:
        if len(self._dataset_class_names) == 0:
            raise ValueError('A dataset must be provided before being able to detect objects.')

        if self._net is None:
            raise ValueError('A neural network must be provided before being able to detect objects.')

        # if we pass no frame, immediately return None
        if frame is None:
            return None

        # if we are not interested in detecting any type of object, we can shortcut the process and return immediately
        if len(self._objects_to_detect) == 0:
            return None

        class_ids, confidences, bounding_boxes = self._net.detect(
            frame = frame,
            confThreshold = self._detection_confidence_threshold,
            nmsThreshold = self._detection_non_maximum_suppression_threshold
        )

        # print('----------------------------------------------')
        # print('class_ids:', class_ids, '\n')
        # print('confidences:', confidences, '\n')
        # print ('bbox:', bounding_boxes, '\n')
        # print('----------------------------------------------\n\n')

        if len(class_ids) == 0:
            return None

        class_names_result = []
        confidences_result = []
        bounding_boxes_result = []

        for class_id, confidence, bounding_box in zip(class_ids.flatten(), confidences.flatten(), bounding_boxes):
            class_name: str = self._dataset_class_names[class_id - 1]

            if class_name in self._objects_to_detect:
                class_names_result.append(class_name)
                confidences_result.append(confidence)
                bounding_boxes_result.append(bounding_box)

        if len(class_names_result) == 0:
            return None

        return class_names_result, confidences_result, bounding_boxes_result
