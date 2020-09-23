import numpy as np
import h5py
from typing import Callable, Any, Dict, List, Tuple
from functools import partial

from .normalizers import rgbNorm, depthNorm, poseNorm, opticalFlowNorm, normalNorm, \
	semanticSegmentationNorm, wireframeNorm, halftoneNorm
from ...h5_dataset_reader import H5DatasetReader, defaultH5DimGetter
from ...internal import DatasetIndex
from ....utilities import smartIndexWrapper

def opticalFlowReader(dataset : h5py._hl.group.Group, index : DatasetIndex, \
	dim : str, readerObj : H5DatasetReader) -> np.ndarray:
	baseDirectory = readerObj.dataset["others"]["baseDirectory"][()]
	paths = dataset[dim][index.start : index.end]

	results = []
	for path in paths:
		path_x, path_y = path
		path_x, path_y = "%s/%s" % (baseDirectory, str(path_x, "utf8")), "%s/%s" % (baseDirectory, str(path_y, "utf8"))
		flow_x, flow_y = readerObj.rawFlowReadFunction(path_x), readerObj.rawFlowReadFunction(path_y)
		flow = np.stack([flow_x, flow_y], axis=-1)
		results.append(flow)
	return np.array(results)

def rgbNeighbourReader(dataset : h5py._hl.group.Group, index : DatasetIndex, \
	skip : int, readerObj : H5DatasetReader) -> np.ndarray:
	baseDirectory = readerObj.dataset["others"]["baseDirectory"][()]

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

def depthReadFunction(path : str, readerObj : H5DatasetReader) -> np.ndarray:
	return readerObj.rawDepthReadFunction(path)

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

class CarlaGenericReader(H5DatasetReader):
	def __init__(self, datasetPath : str, dataBuckets : Dict[str, List[str]], \
		rawReadFunction : Callable[[str], np.ndarray], desiredShape : Tuple[int, int], \
		numNeighboursAhead : int, hyperParameters : Dict[str, Any]):
		assert numNeighboursAhead >= 0

		dimGetter = {
			"rgb" : partial(pathsReader, readerObj=self, readFunction=rawReadFunction, dim="rgb"),
			"depth" : partial(pathsReader, readerObj=self, \
				readFunction=partial(depthReadFunction, readerObj=self), dim="depth"),
			"pose" : partial(defaultH5DimGetter, dim="position"),
			"optical_flow" : partial(opticalFlowReader, readerObj=self),
			"semantic_segmentation" : partial(pathsReader, readerObj=self, readFunction=rawReadFunction, \
				dim="semantic_segmentation"),
			"wireframe" : partial(pathsReader, readerObj=self, readFunction=rawReadFunction, dim="wireframe"),
			"halftone" : partial(pathsReader, readerObj=self, readFunction=rawReadFunction, dim="halftone"),
			"normal" : partial(pathsReader, readerObj=self, readFunction=rawReadFunction, dim="normal"),
			"cameranormal" : partial(pathsReader, readerObj=self, readFunction=rawReadFunction, dim="cameranormal"),
			"rgbDomain2" : partial(pathsReader, readerObj=self, readFunction=rawReadFunction, dim="rgbDomain2"),
		}

		dimTransform ={
			"data" : {
				"rgb" : partial(rgbNorm, readerObj=self),
				"depth" : partial(depthNorm, readerObj=self),
				"pose" : partial(poseNorm, readerObj=self),
				"semantic_segmentation" : partial(semanticSegmentationNorm, readerObj=self),
				"wireframe" : partial(wireframeNorm, readerObj=self),
				"halftone" : partial(halftoneNorm, readerObj=self),
				"normal" : partial(normalNorm, readerObj=self),
				"cameranormal" : partial(normalNorm, readerObj=self),
				"rgbDomain2" : partial(rgbNorm, readerObj=self),
			}
		}

		for i in range(numNeighboursAhead):
			key = "rgbNeighbour(t+%d)" % (i + 1)
			dimGetter[key] = partial(rgbNeighbourReader, skip=i + 1, readerObj=self)
			dimTransform["data"][key] = partial(rgbNorm, readerObj=self)

			flowKey = "optical_flow(t+%d)" % (i + 1)
			dimGetter[flowKey] = partial(opticalFlowReader, readerObj=self, dim=flowKey)
			dimTransform["data"][flowKey] = partial(opticalFlowNorm, readerObj=self)

		super().__init__(datasetPath, dataBuckets, dimGetter, dimTransform)
		self.rawReadFunction = rawReadFunction
		self.numNeighboursAhead = numNeighboursAhead
		self.idOfNeighbour = self.getIdsOfNeighbour()
		self.desiredShape = desiredShape
		self.hyperParameters = hyperParameters

	# For each top level (train/tet/val) create a new array with the index of the frame at time t + skipFrames.
	# For example result["train"][0] = 2550 means that, after randomization the frame at time=1 is at id 2550.
	def getIdsOfNeighbour(self):
		def f(ids, skipFrames):
			N = len(ids)
			closest = np.zeros(N, dtype=np.uint32)
			for i in range(N):
				where = np.where(ids == ids[i] + skipFrames)[0]
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
			result[topLevel] = {}
			# Store the ids up to the number of neighbours we are interested in (for optical flow with skip mostly)
			for i in range(self.numNeighboursAhead):
				key = "t+%d" % (i + 1)
				result[topLevel][key] = f(ids, i + 1)
		return result

	def __str__(self) -> str:
		return "[CarlaH5PathsNpyReader] H5 File: %s" % (self.datasetPath)
