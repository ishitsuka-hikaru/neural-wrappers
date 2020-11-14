import h5py
import numpy as np
from typing import Dict, List, Callable
from overrides import overrides
from .dataset_reader import DatasetReader, DimGetterCallable
from .internal import DatasetIndex, DatasetRange, DatasetRandomIndex
from ..utilities import isType, flattenList, smartIndexWrapper
from returns.curry import partial

def defaultH5DimGetter(dataset : h5py._hl.group.Group, index : DatasetIndex, dim : str):
	if isType(index, DatasetRange):
		return dataset[dim][index.start : index.end][()] #type: ignore
	elif isType(index, np.ndarray):
		return smartIndexWrapper(dataset[dim], index)
	assert False

class H5DatasetReader(DatasetReader):
	def __init__(self, datasetPath : str, dataBuckets : Dict[str, List[str]], \
		dimGetter : Dict[str, DimGetterCallable], dimTransform : Dict[str, Dict[str, Callable]]):
		self.datasetPath = datasetPath
		self.dataset = h5py.File(self.datasetPath, "r")
		super().__init__(dataBuckets, dimGetter, dimTransform)

	def sanitizeDimGetter(self, dimGetter : Dict[str, Callable]) -> Dict[str, Callable]:
		allDims : List[str] = list(set(flattenList(self.dataBuckets.values())))
		for key in allDims:
			if not key in dimGetter:
				print("[H5DatasetReader::sanitizeDimGetter] Adding default dimGetter for '%s'" % key)
				dimGetter[key] = partial(defaultH5DimGetter, dim=key)
		return dimGetter

	def getDataset(self, topLevel : str) -> h5py._hl.group.Group:
		return self.dataset[topLevel]

	def getBatchDatasetIndex(self, i : int, topLevel : str, batchSize : int) -> DatasetIndex:
		startIndex = i * batchSize
		endIndex = min((i + 1) * batchSize, self.getNumData(topLevel))
		assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)
		return DatasetRange(startIndex, endIndex)

	@overrides
	def getNumData(self, topLevel:str) -> int:
		firstKey = tuple(self.dimGetter.keys())[0]
		return self.getDataset(topLevel)[firstKey].shape[0]
