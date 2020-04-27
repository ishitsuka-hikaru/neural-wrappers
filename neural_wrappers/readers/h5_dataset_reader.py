import h5py
from functools import partial
from typing import Dict, List, Callable
from .dataset_reader import DatasetReader, DimGetterCallable
from .internal import DatasetIndex, DatasetRange, DatasetRandomIndex
from ..utilities import isType, flattenList

def defaultH5DimGetter(dim : str, index : DatasetIndex, dataset : h5py._hl.files.File):
	if isType(index, DatasetRange):
		return dataset[dim][index.start : index.end] #type: ignore
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
				dimGetter[key] = partial(defaultH5DimGetter, dataset=self.dataset)
		return dimGetter
