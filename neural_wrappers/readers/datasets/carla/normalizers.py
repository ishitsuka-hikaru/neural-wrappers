import numpy as np
from ....utilities import resize_batch, h5ReadDict
from typing import Dict, Union
from ...dataset_reader import DatasetReader

# TODO, [0 : 1] and [-1 : 1] as well.
def rgbNorm(x : np.ndarray) -> np.ndarray:
	# x [MBx854x854x3] => [MBx256x256x3] :: [0 : 255]
	x = resize_batch(x, height=256, width=256)
	# x :: [0 : 255] => [0: 1]
	x = x.astype(np.float32) / 255
	return x

# TODO, [0 : 1] and [-1 : 1] as well.
def depthNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	depthStats = h5ReadDict(readerObj.dataset["others"]["dataStatistics"]["depth"])
	x = np.clip(x, depthStats["min"], depthStats["max"])
	x = (x - depthStats["min"]) / (depthStats["max"] - depthStats["min"])
	return x

# TODO, [0 : 1] and [-1 : 1] as well.
def positionNorm(x : np.ndarray, readerObj : Union[DatasetReader]) -> np.ndarray:
	positionStats = h5ReadDict(readerObj.dataset["others"]["dataStatistics"]["position"])

	minPos, maxPos = positionStats["min"][0 : 3], positionStats["max"][0 : 3]
	translation, rotation = x[:, 0 : 3], x[:, 3 :]
	# Now, just for [0 : 1]
	translation = (translation - minPos) / (maxPos - minPos)
	rotation = ((rotation / 180) + 1) / 2
	position = np.concatenate([translation, rotation], axis=-1)

	return position