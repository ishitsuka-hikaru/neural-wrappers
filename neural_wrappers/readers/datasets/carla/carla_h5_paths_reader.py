import h5py
import numpy as np
from typing import Dict, Callable, List
from functools import partial
from ...internal import DatasetIndex
from ...h5_dataset_reader import H5DatasetReader, defaultH5DimGetter
from ....utilities import tryReadImage, smartIndexWrapper

from .normalizers import rgbNorm, depthNorm, positionNorm
from .utils import unrealFloatFromPng

class CarlaH5PathsReader(H5DatasetReader):
	def __init__(self, datasetPath : str, dataBuckets : Dict[str, List[str]]):
		dimGetter = {
			"rgb" : partial(pathsReader, readerObj=self, readFunction=rgbReader, dim="rgb"),
			"depth" : partial(pathsReader, readerObj=self, readFunction=depthReader, dim="depth"),
			"position" : partial(defaultH5DimGetter, dim="position"),
			"optical_flow" : partial(opticalFlowReader, readerObj=self)
		}

		dimTransform ={
			"data" : {
				"rgb" : rgbNorm,
				"depth" : partial(depthNorm, readerObj=self),
				"position" : partial(positionNorm, readerObj=self)
			},
			"labels" : {
				"position" : lambda x : x
			}
		}
		super().__init__(datasetPath, dataBuckets, dimGetter, dimTransform)

		self.idOfNeighbour = self.getIdsOfNeighbour()

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
	return tryReadImage(path).astype(np.uint8)

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
		# Move the flow in [-1 : 1] from [0 : 1]
		flow = (flow - 0.5) * 2
		results.append(flow)
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