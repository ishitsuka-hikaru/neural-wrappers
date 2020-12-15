import h5py
import numpy as np
from overrides import overrides
from returns.curry import partial
from .batched_dataset_reader import BatchedDatasetReader, DimGetterCallable
from ..internal import DatasetIndex, DatasetRange
from ...utilities import isType, flattenList, smartIndexWrapper

def defaultH5DimGetter(dataset : h5py._hl.group.Group, index : DatasetIndex, dim : str):
	if isType(index, DatasetRange):
		return dataset[dim][index.start : index.end][()] #type: ignore
	elif isType(index, range):
		return dataset[dim][index.start : index.stop][()]
	elif isType(index, np.ndarray) or isType(index, list) or isType(index, tuple):
		return smartIndexWrapper(dataset[dim], index)
	assert False

class H5DatasetReader(BatchedDatasetReader):
	def __init__(self, datasetPath:str, dataBuckets, dimTransform):
		self.datasetPath = datasetPath
		self.dataset = h5py.File(self.datasetPath, "r")

		allDims = list(set(flattenList(dataBuckets.values())))
		dimGetter = {k : partial(defaultH5DimGetter, dim=k) for k in allDims}
		super().__init__(dataBuckets, dimGetter, dimTransform)

	@overrides
	def getDataset(self, topLevel : str) -> h5py._hl.group.Group:
		return self.dataset[topLevel]

	@overrides
	def getNumData(self, topLevel:str) -> int:
		firstKey = tuple(self.dimGetter.keys())[0]
		return len(self.getDataset(topLevel)[firstKey])
