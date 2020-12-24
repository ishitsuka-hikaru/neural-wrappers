import numpy as np
import transforms3d.euler as txe
import h5py
from typing import Callable

from .carla_h5_paths_reader import CarlaH5PathsReader
from ....utilities import npCloseEnough, npGetInfo, smartIndexWrapper

def unrealFloatFromPng(x : np.ndarray) -> np.ndarray:
	x = x.astype(np.float32)
	x = (x[..., 0] + x[..., 1] * 256 + x[..., 2] * 256 * 256) / (256 * 256 * 256 - 1)
	x = x.astype(np.float32)
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

# def opticalFlowReader(dataset:h5py._hl.group.Group, index:range, \
# 	dim:str, rawReadFunction:Callable[[str], np.ndarray]) -> np.ndarray:
# 	baseDirectory = dataset.file["others"]["baseDirectory"][()]
# 	paths = dataset[dim][index.start : index.stop]
# 	breakpoint()

# 	results = []
# 	for path in paths:
# 		path_x, path_y = path
# 		path_x, path_y = "%s/%s" % (baseDirectory, str(path_x, "utf8")), "%s/%s" % (baseDirectory, str(path_y, "utf8"))
# 		flow_x, flow_y = readerObj.rawFlowReadFunction(path_x), readerObj.rawFlowReadFunction(path_y)
# 		flow = np.stack([flow_x, flow_y], axis=-1)
# 		results.append(flow)
# 	return np.array(results)

# def rgbNeighbourReader(dataset:h5py._hl.group.Group, index:range, \
# 	skip:int, readerObj:CarlaH5PathsReader) -> np.ndarray:
# 	baseDirectory = dataset.file["others"]["baseDirectory"][()]

# 	# For optical flow we have the problem that the flow data for t->t+1 is stored at index t+1, which isn't
# 	#  necessarily 1 index to the right (trian set may be randomized beforehand). Thus, we need to get the indexes
# 	#  of the next neighbours of this top level (train/test etc.), and then read the paths at those indexes.
# 	topLevel = readerObj.getActiveTopLevel()
# 	key = "t+%d" % (skip)
# 	neighbourIds = readerObj.idOfNeighbour[topLevel][key][index.start : index.stop]
# 	paths = smartIndexWrapper(dataset["rgb"], neighbourIds)

# 	results = []
# 	for path in paths:
# 		path = "%s/%s" % (baseDirectory, str(path, "utf8"))
# 		results.append(readerObj.rawReadFunction(path))
# 	return np.array(results)

# def depthReadFunction(path:str, readerObj:CarlaH5PathsReader) -> np.ndarray:
# 	return readerObj.rawDepthReadFunction(path)

# Append base directory to all paths read from the h5, and then call the reading function for each full path.
def pathsReader(dataset:h5py._hl.group.Group, index:np.ndarray, \
	readFunction:Callable[[str], np.ndarray], dim:str) -> np.ndarray:
	baseDirectory = dataset.file["others"]["baseDirectory"][()]
	try:
		baseDirectory = str(baseDirectory, "utf8")
	except Exception:
		pass

	try:
		paths = dataset[dim][index]
	except Exception:
		paths = smartIndexWrapper(dataset[dim], index)

	results = []
	for path in paths:
		path = "%s/%s" % (baseDirectory, str(path, "utf8"))
		results.append(readFunction(path))
	return np.array(results)
