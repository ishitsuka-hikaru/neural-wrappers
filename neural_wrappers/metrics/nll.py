import numpy as np

# @brief negative log likelihood (cross entropy metric)
# @param[in] y Predictions. Shape :: MBx(Shape)xNC with positive values
# @param[in] t Labels. Shape :: MBx(Shape)xNC with binary values
# @return The NLL result for each channel. SHape :: MBx(Shape)
def nll(y, t):
	t = t.max(axis=-1, keepdims=True) == t
	t = t.astype(np.int32)
	L = (y * t).max(axis=-1)
	L = -np.log(L)
	L[~np.isfinite(L)] = 100
	return L
