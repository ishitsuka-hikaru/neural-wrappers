import numpy as np
from typing import Dict, Union
from .utils import getQuatFromRotation
from ....utilities import resize_batch, h5ReadDict
from ...dataset_reader import DatasetReader

# TODO: All norms now put data in [0 : 1]. We should look at the rederObj and if some dims want other range, transform
#  the data to that range.

def rgbNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	# x [MBx854x854x3] => [MBx256x256x3] :: [0 : 255]
	x = resize_batch(x, height=256, width=256)
	# x :: [0 : 255] => [0: 1]
	x = x.astype(np.float32) / 255
	return x

def depthNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	depthStats = h5ReadDict(readerObj.dataset["others"]["dataStatistics"]["depth"])
	x = resize_batch(x, height=256, width=256)
	x = np.clip(x, depthStats["min"], depthStats["max"])
	x = (x - depthStats["min"]) / (depthStats["max"] - depthStats["min"])
	return x

def positionNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	positionStats = h5ReadDict(readerObj.dataset["others"]["dataStatistics"]["position"])

	minPos, maxPos = positionStats["min"][0 : 3], positionStats["max"][0 : 3]
	translation, rotation = x[:, 0 : 3], x[:, 3 :]
	# Now, just for [0 : 1]
	translation = (translation - minPos) / (maxPos - minPos)
	rotation = ((rotation / 180) + 1) / 2
	position = np.concatenate([translation, rotation], axis=-1)

	return position

def positionQuatNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	positionStats = h5ReadDict(readerObj.dataset["others"]["dataStatistics"]["position"])

	minPos, maxPos = positionStats["min"][0 : 3], positionStats["max"][0 : 3]
	translation, rotation = x[:, 0 : 3], x[:, 3 :]
	# Now, just for [0 : 1]
	# Translation is easy, just min max it
	translation = (translation - minPos) / (maxPos - minPos)

	# Rotation is in [-180 : 180], move to [-1 : 1], then call getQuatFromRotation
	rotation = rotation / 180
	quatRotation = getQuatFromRotation(rotation)
	# The returned quaternion is in [-1 : 1], we move it to [0 : 1]
	quatRotation = (quatRotation + 1) / 2
	position = np.concatenate([translation, quatRotation], axis=-1)

	return position

def opticalFlowNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	x = resize_batch(x, height=256, width=256)
	return x

def normalNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	x = resize_batch(x, height=256, width=256)
	# Normals are stored as [0 - 255] on 3 channels, representing orientation of the 3 axes.
	x = x.astype(np.float32) / 255
	return x

def semanticSegmentationNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	x = resize_batch(x, height=256, width=256, interpolation="nearest")
	return x