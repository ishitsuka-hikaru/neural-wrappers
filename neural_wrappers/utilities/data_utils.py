from typing import Optional
from .type_utils import NWNumber
import numpy as np

def minMax(x:np.ndarray, Min:Optional[NWNumber]=None, Max:Optional[NWNumber]=None):
	x = x.astype(np.float32)
	if Min is None:
		Min = x.min()
	if Max is None:
		Max = x.max()
	return (x - Min) / (Max - Min + np.spacing(1))

def minMaxPercentile(x, low=0, high=100):
	Min, Max = np.percentile(x, [low, high])
	x = np.clip(x, Min, Max)
	return minMax(x)

def standardize(x, Mean:Optional[NWNumber]=None, Std:Optional[NWNumber]=None):
	x = x.astype(np.float32)
	if Mean is None:
		Mean = x.mean()
	if Std is None:
		Std = x.std()
	return (x - Mean) / (Std + np.spacing(1))

def standardizePercentile(x, low=0, high=100):
	Min, Max = np.percentile(x, [low, high])
	x = np.clip(x, Min, Max)
	return standardize(x)

def toCategorical(data, numClasses):
	data = np.array(data)
	y = np.eye(numClasses)[data.reshape(-1)].astype(np.uint8)
	# Some bugs for (1, 1) shapes return (1, ) instead of (1, NC)
	MB = data.shape[0]
	y = np.squeeze(y)
	if MB == 1:
		y = np.expand_dims(y, axis=0)
	y = y.reshape(*data.shape, numClasses)
	return y
