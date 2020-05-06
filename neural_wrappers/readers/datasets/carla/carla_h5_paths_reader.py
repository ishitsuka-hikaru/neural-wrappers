import h5py
import numpy as np
from typing import Dict, Callable, List, Tuple, Any
from functools import partial
from ...internal import DatasetIndex
from ...h5_dataset_reader import H5DatasetReader, defaultH5DimGetter
from ....utilities import tryReadImage, smartIndexWrapper, toCategorical

from .normalizers import rgbNorm, depthNorm, positionNorm, opticalFlowNorm, \
	normalNorm, semanticSegmentationNorm, positionQuatNorm
from .utils import unrealFloatFromPng

class CarlaH5PathsReader(H5DatasetReader):
	def __init__(self, datasetPath : str, dataBuckets : Dict[str, List[str]], \
	desiredShape : Tuple[int, int], hyperParameters : Dict[str, Any]):
		dimGetter = {
			"rgb" : partial(pathsReader, readerObj=self, readFunction=rgbReader, dim="rgb"),
			"depth" : partial(pathsReader, readerObj=self, readFunction=depthReader, dim="depth"),
			"position" : partial(defaultH5DimGetter, dim="position"),
			"optical_flow" : partial(opticalFlowReader, readerObj=self),
			"semantic_segmentation" : partial(pathsReader, readerObj=self, readFunction=semanticSegmentationReader, \
				dim="semantic_segmentation"),
			"wireframe" : partial(pathsReader, readerObj=self, readFunction=rgbReader, dim="wireframe"),
			"halftone" : partial(pathsReader, readerObj=self, readFunction=rgbReader, dim="halftone"),
			"normal" : partial(pathsReader, readerObj=self, readFunction=rgbReader, dim="normal"),
			"cameranormal" : partial(pathsReader, readerObj=self, readFunction=rgbReader, dim="cameranormal"),
			"rgbDomain2" : partial(pathsReader, readerObj=self, readFunction=rgbReader, dim="rgbDomain2"),
			"rgbNeighbour" : partial(rgbNeighbourReader, readerObj=self),
			"position_quat" : partial(defaultH5DimGetter, dim="position"),
		}

		dimTransform ={
			"data" : {
				"rgb" : partial(rgbNorm, readerObj=self),
				"depth" : partial(depthNorm, readerObj=self),
				"position" : partial(positionNorm, readerObj=self),
				"optical_flow" : partial(opticalFlowNorm, readerObj=self),
				"semantic_segmentation" : partial(semanticSegmentationNorm, readerObj=self),
				"wireframe" : partial(rgbNorm, readerObj=self),
				"halftone" : partial(rgbNorm, readerObj=self),
				"normal" : partial(normalNorm, readerObj=self),
				"cameranormal" : partial(normalNorm, readerObj=self),
				"rgbDomain2" : partial(rgbNorm, readerObj=self),
				"rgbNeighbour" : partial(rgbNorm, readerObj=self),
				"position_quat" : partial(positionQuatNorm, readerObj=self)
			}
		}
		super().__init__(datasetPath, dataBuckets, dimGetter, dimTransform)

		self.idOfNeighbour = self.getIdsOfNeighbour()
		self.desiredShape = desiredShape
		self.hyperParameters = hyperParameters

	# For each top level (train/tet/val) create a new array with the index of the frame at time t + 1.
	# For example result["train"][0] = 2550 means that, after randomization the frame at time=1 is at id 2550.
	def getIdsOfNeighbour(self):
		def f(ids):
			N = len(ids)
			closest = np.zeros(N, dtype=np.uint32)
			for i in range(N):
				where = np.where(ids == ids[i] + 1)[0]
				if len(where) == 0:
					where = [i]
				assert len(where) == 1
				closest[i] = where[0]
			return closest

		result = {}
		for topLevel in self.dataset.keys():
			# Just top levels with "ids" key are valid. There may be "others" or some other top level keys as well.
			if not "ids" in self.dataset[topLevel]:
				continue
			ids = self.dataset[topLevel]["ids"][()]
			neighbours = f(ids)
			result[topLevel] = neighbours
		return result

	def __str__(self) -> str:
		return "[CarlaH5PathsReader] H5 File: %s" % (self.datasetPath)

def rgbReader(path : str) -> np.ndarray:
	return tryReadImage(path)

def depthReader(path : str) -> np.ndarray:
	dph = tryReadImage(path)
	dph = unrealFloatFromPng(dph) * 1000
	return np.expand_dims(dph, axis=-1)

def opticalFlowReader(dataset : h5py._hl.group.Group, index : DatasetIndex, readerObj : H5DatasetReader) -> np.ndarray:
	baseDirectory = readerObj.dataset["others"]["baseDirectory"][()]

	# For optical flow we have the problem that the flow data for t->t+1 is stored at index t+1, which isn't
	#  necessarily 1 index to the right (trian set may be randomized beforehand). Thus, we need to get the indexes
	#  of the next neighbours of this top level (train/test etc.), and then read the paths at those indexes.
	topLevel = readerObj.getActiveTopLevel()
	neighbourIds = readerObj.idOfNeighbour[topLevel][index.start : index.end]
	paths = smartIndexWrapper(dataset["optical_flow"], neighbourIds)

	# Also, there are two optical flow images for each index, so we need to read both.
	results = []
	for path in paths:
		path_x, path_y = path
		path_x, path_y = "%s/%s" % (baseDirectory, str(path_x, "utf8")), "%s/%s" % (baseDirectory, str(path_y, "utf8"))
		flow_x, flow_y = unrealFloatFromPng(tryReadImage(path_x)), unrealFloatFromPng(tryReadImage(path_y))
		flow = np.array([flow_x, flow_y]).transpose(1, 2, 0)
		results.append(flow)
	return np.array(results)

def semanticSegmentationReader(path : str) -> np.ndarray:
	labelKeys = list({
		(0, 0, 0): "Unlabeled",
		(70, 70, 70): "Building",
		(153, 153, 190): "Fence",
		(160, 170, 250): "Other",
		(60, 20, 220): "Pedestrian",
		(153, 153, 153): "Pole",
		(50, 234, 157): "Road line",
		(128, 64, 128): "Road",
		(232, 35, 244): "Sidewalk",
		(35, 142, 107): "Vegetation",
		(142, 0, 0): "Car",
		(156, 102, 102): "Wall",
		(0, 220, 220): "Traffic sign"
	}.keys())
	item = tryReadImage(path).astype(np.uint32)

	newItem = item[..., 0] + item[..., 1] * 256 + item[..., 2] * 256 * 256
	labelKeys = list(map(lambda x : x[0] + x[1] * 256 + x[2] * 256 * 256, labelKeys))
	for i in range(len(labelKeys)):
		newItem[newItem == labelKeys[i]] = i
	return np.eye(13)[newItem].astype(np.float32)

def rgbNeighbourReader(dataset : h5py._hl.group.Group, index : DatasetIndex, \
	readerObj : H5DatasetReader) -> np.ndarray:
	baseDirectory = readerObj.dataset["others"]["baseDirectory"][()]

	# For optical flow we have the problem that the flow data for t->t+1 is stored at index t+1, which isn't
	#  necessarily 1 index to the right (trian set may be randomized beforehand). Thus, we need to get the indexes
	#  of the next neighbours of this top level (train/test etc.), and then read the paths at those indexes.
	topLevel = readerObj.getActiveTopLevel()
	neighbourIds = readerObj.idOfNeighbour[topLevel][index.start : index.end]
	paths = smartIndexWrapper(dataset["rgb"], neighbourIds)

	results = []
	for path in paths:
		path = "%s/%s" % (baseDirectory, str(path, "utf8"))
		results.append(rgbReader(path))
	return np.array(results)

# Append base directory to all paths read from the h5, and then call the reading function for each full path.
def pathsReader(dataset : h5py._hl.group.Group, index : DatasetIndex, readerObj : H5DatasetReader,
	readFunction : Callable[[str], np.ndarray], dim : str) -> np.ndarray:
	baseDirectory = readerObj.dataset["others"]["baseDirectory"][()]
	paths = dataset[dim][index.start : index.end]

	results = []
	for path in paths:
		path = "%s/%s" % (baseDirectory, str(path, "utf8"))
		results.append(readFunction(path))
	return np.array(results)