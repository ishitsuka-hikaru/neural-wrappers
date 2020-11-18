import numpy as np
import transforms3d.euler as txe
import h5py
from typing import Callable

from .carla_generic_reader import CarlaGenericReader
from ....utilities import npCloseEnough, npGetInfo, smartIndexWrapper
from ...internal import DatasetRange
from ...h5_dataset_reader import H5DatasetReader

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

def opticalFlowReader(dataset:h5py._hl.group.Group, index:DatasetRange, \
	dim:str, readerObj:CarlaGenericReader) -> np.ndarray:
	baseDirectory = str(readerObj.dataset["others"]["baseDirectory"][()], "utf8")
	paths = dataset[dim][index.start : index.end]

	results = []
	for path in paths:
		path_x, path_y = path
		path_x, path_y = "%s/%s" % (baseDirectory, str(path_x, "utf8")), "%s/%s" % (baseDirectory, str(path_y, "utf8"))
		flow_x, flow_y = readerObj.rawFlowReadFunction(path_x), readerObj.rawFlowReadFunction(path_y)
		flow = np.stack([flow_x, flow_y], axis=-1)
		results.append(flow)
	return np.array(results)

def rgbNeighbourReader(dataset:h5py._hl.group.Group, index:DatasetRange, \
	skip:int, readerObj:CarlaGenericReader) -> np.ndarray:
	baseDirectory = str(readerObj.dataset["others"]["baseDirectory"][()], "utf8")

	# For optical flow we have the problem that the flow data for t->t+1 is stored at index t+1, which isn't
	#  necessarily 1 index to the right (trian set may be randomized beforehand). Thus, we need to get the indexes
	#  of the next neighbours of this top level (train/test etc.), and then read the paths at those indexes.
	topLevel = readerObj.getActiveTopLevel()
	key = "t+%d" % (skip)
	neighbourIds = readerObj.idOfNeighbour[topLevel][key][index.start : index.end]
	paths = smartIndexWrapper(dataset["rgb"], neighbourIds)

	results = []
	for path in paths:
		path = "%s/%s" % (baseDirectory, str(path, "utf8"))
		results.append(readerObj.rawReadFunction(path))
	return np.array(results)

def depthReadFunction(path:str, readerObj:CarlaGenericReader) -> np.ndarray:
	return readerObj.rawDepthReadFunction(path)

# Append base directory to all paths read from the h5, and then call the reading function for each full path.
def pathsReader(dataset : h5py._hl.group.Group, index : DatasetRange, readerObj:CarlaGenericReader,
	readFunction : Callable[[str], np.ndarray], dim:str) -> np.ndarray:
	baseDirectory = str(readerObj.dataset["others"]["baseDirectory"][()], "utf8")
	paths = dataset[dim][index.start : index.end]

	results = []
	for path in paths:
		path = "%s/%s" % (baseDirectory, str(path, "utf8"))
		results.append(readFunction(path))
	return np.array(results)
