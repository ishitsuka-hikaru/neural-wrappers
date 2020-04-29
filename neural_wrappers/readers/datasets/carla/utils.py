import numpy as np

def unrealFloatFromPng(x : np.ndarray) -> np.ndarray:
	x = (x[..., 0] + x[..., 1] * 256 + x[..., 2] * 256 * 256) / (256 * 256 * 256 - 1)
	return x.astype(np.float32)
