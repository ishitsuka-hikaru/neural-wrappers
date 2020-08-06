import numpy as np
from typing import Union
from ....utilities import resize_batch
from ...dataset_reader import DatasetReader

def rgbNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	# x [MBx480x640x3] => [MBxHxWx3] :: [0 : 255]
	x = resize_batch(x, height=readerObj.desiredShape[0], width=readerObj.desiredShape[1], resizeLib="opencv")
	# x :: [0 : 255] => [0: 1]
	x = x.astype(np.float32) / 255
	return x

def depthNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	depthStats = {"min" : 0, "max" : readerObj.dataset["others"]["maxDepthMeters"][()]}

	x = resize_batch(x, height=readerObj.desiredShape[0], width=readerObj.desiredShape[1], resizeLib="opencv")
	# Depth is stored in [0 : 1] representing up to 1000m from simulator
	x = np.clip(x, depthStats["min"], depthStats["max"])
	x = (x - depthStats["min"]) / (depthStats["max"] - depthStats["min"])
	return np.expand_dims(x, axis=-1)

def normalNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	x = resize_batch(x, height=readerObj.desiredShape[0], width=readerObj.desiredShape[1], resizeLib="opencv")
	# Normals are stored as [0 - 255] on 3 channels, representing orientation of the 3 axes.
	x = x.astype(np.float32) / 255
	return x

def semanticSegmentationNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	x = x.astype(np.uint8)
	x = resize_batch(x, interpolation="nearest", height=readerObj.desiredShape[0], \
		width=readerObj.desiredShape[1], resizeLib="opencv")

	x = x[..., 0]
	x[x == 255] = 40
	numClasses = 41

	# Some fancy way of doing one-hot encoding.
	x = np.eye(numClasses)[x].astype(np.float32)
	return x