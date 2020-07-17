import numpy as np
from typing import Tuple

def computeIntrinsicMatrix(fieldOfView : int, resolution : Tuple[int, int], skew:float = 0):
	cy = resolution[0] / 2
	cx = resolution[1] / 2
	fy = cy / (np.tan(fieldOfView * np.pi / 360))
	fx = cx / (np.tan(fieldOfView * np.pi / 360))

	K = np.array([
		[fx, skew, cx],
		[0, fy, cy],
		[0, 0, 1]
	], dtype=np.float32)
	return K
