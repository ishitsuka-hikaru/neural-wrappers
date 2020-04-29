import numpy as np
import transforms3d.euler as txe

def unrealFloatFromPng(x : np.ndarray) -> np.ndarray:
	x = (x[..., 0] + x[..., 1] * 256 + x[..., 2] * 256 * 256) / (256 * 256 * 256 - 1)
	return x.astype(np.float32)

# Given a rotation in [-1 : 1] represnting [-180 : +180], we transform it to a quaternion representation (batched)
def getQuatFromRotation(rotation : np.ndarray) -> np.ndarray:
	assert rotation.min() >= -1 and rotation.max() <= 1
	# Move rotation from [-1 : 1] to [0 : 2*pi]
	roll, pitch, yaw = (rotation.T + 1) * np.pi
	quat = np.float32(np.concatenate([[txe.euler2quat(roll[i], pitch[i], yaw[i])] for i in range(len(roll))]))
	return quat
