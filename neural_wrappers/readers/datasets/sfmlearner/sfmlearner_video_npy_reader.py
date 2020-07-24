import numpy as np
import h5py
from typing import Dict, Callable, Any, List
from overrides import overrides
from functools import partial

from .sfmlearner_generic_reader import SfmLearnerGenericReader
from ...internal import DatasetRandomIndex, DatasetIndex
from ....utilities import smartIndexWrapper

def defaultRgbGetter(dataset, index:DatasetRandomIndex, baseDirectory:str):
	# TODO: Fix smartIndexWrapper for str paths
	items = smartIndexWrapper(dataset["data"]["rgb"], index.sequence, f = lambda data, index : np.array([data[index]]))
	result = []

	for item in items.flatten():
		path = "%s/%s" % (baseDirectory, item)
		npyItem = np.load(path)
		result.append(npyItem)
	result = np.array(result, dtype=np.uint8).reshape(*items.shape[0 : 2], *npyItem.shape)
	return result

class SfmLearnerVideoNpyReader(SfmLearnerGenericReader):
	def __init__(self, datasetPath:str, sequenceSize:int, dataSplitIndices:Dict[str, List[int]], \
		intrinsicMatrix:np.ndarray = np.eye(3), dimTransform:Dict[str, Dict[str, Callable]]={}):

		self.datasetPath = datasetPath
		self.dataset = h5py.File(self.datasetPath, "r")
		self.fps = self.dataset["others"]["fps"][()]
		self.frameShape = self.dataset["others"]["resolution"][()]
		self.dataSplitIndices = dataSplitIndices

		rgbGetter = partial(defaultRgbGetter, baseDirectory=self.dataset["others"]["baseDirectory"][()])
		super().__init__(
			dataBuckets = {"data" : ["rgb", "intrinsics"]}, \
			dimGetter = {"rgb" : rgbGetter, "intrinsics" : (lambda _, __ : intrinsicMatrix)}, \
			dimTransform = dimTransform,
			dataSplitIndices = dataSplitIndices,
			intrinsicMatrix = intrinsicMatrix
		)

	@overrides
	def getNumData(self, topLevel:str) -> int:
		return len(self.dataSplitIndices[topLevel])

	@overrides
	def getDataset(self, topLevel:str) -> Any:
		return self.dataset

	@overrides
	def getBatchDatasetIndex(self, i:int, topLevel:str, batchSize:int) -> DatasetIndex:
		startIndex = i * batchSize
		endIndex = min((i + 1) * batchSize, self.getNumData(topLevel))
		indices = self.dataSplitIndices[topLevel][startIndex : endIndex]
		return DatasetRandomIndex(indices)

	def __str__(self) -> str:
		Str = "[SfmLearnerVideoNpyReader]"
		Str += "\n - Path: %s" % (self.datasetPath)
		Str += "\n - Resolution: %d x %d" % (self.frameShape[0], self.frameShape[1])
		Str += "\n - Num frames: %d. FPS: %2.3f" % (len(self.dataset["data"]["rgb"]), self.fps)
		Str += "\n - Sequence size: %d. Skip frames: %d." % (self.sequenceSize, self.skipFrames)
		Str += "\n - Intrinsic camera: %s" % (self.intrinsicMatrix.tolist())
		Str += "\n - Data splits: %s" % (list(self.dataSplitIndices.keys()))
		Str += "\n - Data split counts: %s" % ({k : len(self.dataSplitIndices[k]) for k in self.dataSplitIndices})
		return Str

