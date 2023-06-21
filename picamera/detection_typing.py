import numpy as np

Frame = np.ndarray
BoundingBox = tuple[int, int, int, int]
ObjectDetectionResult = list[tuple[str, float, BoundingBox]]
Point = tuple[int, int]
