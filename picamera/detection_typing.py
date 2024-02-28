from typing import Union
import numpy as np

Frame = np.ndarray
BoundingBox = Union[list[int], tuple[int, int, int, int]]
ObjectDetectionResult = list[tuple[str, float, BoundingBox]]
Point = tuple[int, int]
