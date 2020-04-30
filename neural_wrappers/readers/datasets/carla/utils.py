import numpy as np
import transforms3d.euler as txe
from ....utilities import npCloseEnough, npGetInfo

def unrealFloatFromPng(x : np.ndarray) -> np.ndarray:
	x = (x[..., 0] + x[..., 1] * 256 + x[..., 2] * 256 * 256) / (256 * 256 * 256 - 1)
	x = np.float32(x)
	return x

def unrealPngFromFloat(x : np.ndarray) -> np.ndarray:
	assert x.dtype == np.float32
	y = np.int32(x * (256 * 256 * 256 + 1))
	# Shrink any additional bits outside of 24 bits
	y = y & (256 * 256 * 256 - 1)
	R = y & 255
	G = (y >> 8) & 255
	B = (y >> 16) & 255
	result = np.array([R, G, B], dtype=np.uint8).transpose(1, 2, 0)
	assert npCloseEnough(x, unrealFloatFromPng(result), eps=1e-2)
	return result

# Given a rotation in [-1 : 1] represnting [-180 : +180], we transform it to a quaternion representation (batched)
def getQuatFromRotation(rotation : np.ndarray) -> np.ndarray:
	assert rotation.min() >= -1 and rotation.max() <= 1
	# Move rotation from [-1 : 1] to [0 : 2*pi]
	roll, pitch, yaw = (rotation.T + 1) * np.pi
	quat = np.float32(np.concatenate([[txe.euler2quat(roll[i], pitch[i], yaw[i])] for i in range(len(roll))]))
	return quat
