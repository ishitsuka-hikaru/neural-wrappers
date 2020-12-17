import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any, Iterator, Union, Tuple
from prefetch_generator import BackgroundGenerator
from copy import deepcopy

from .dataset_types import DimGetterCallable, DimTransformCallable, DatasetIndex, Item
from ..utilities import flattenList

# @param[in] dataBuckets A dictionary with all available data bucket names (data, label etc.) and, for each bucket,
#  a list of dimensions (rgb, depth, etc.).
#  Example: {"data":["rgb", "depth"], "labels":["depth", "semantic"]}
# @param[in] dimGetter For each possible dimension defined above, we need to receive a method that tells us how
#  to retrieve a batch of items. Some dimensions may be overlapped in multiple data bucket names, however, they are
#  logically the same information before transforms, so we only read it once and copy in memory if needed.
# @param[in] dimTransform The transformations for each dimension of each topdata bucket name. Some dimensions may
#  overlap and if this happens we duplicate the data to ensure consistency. This may be needed for cases where
#  the same dimension may be required in 2 formats (i.e. position as quaternions as well as unnormalized 6DoF).
class DatasetFormat:
	def __init__(self, dataBuckets:Dict[str, List[str]], dimGetter:Dict[str, DimGetterCallable], \
		dimTransform:Dict[str, Dict[str, DimTransformCallable]]):
		self.allDims = list(set(flattenList(dataBuckets.values())))
		self.dataBuckets = dataBuckets
		self.dimGetter = self.sanitizeDimGetter(dimGetter)
		self.dimTransform = self.sanitizeDimTransform(dimTransform)

		self.dimToDataBuckets = {dim : [] for dim in self.allDims}
		for dim in self.allDims:
			for bucket in self.dataBuckets:
				if dim in self.dataBuckets[bucket]:
					self.dimToDataBuckets[dim].append(bucket)

	def sanitizeDimGetter(self, dimGetter:Dict[str, Callable]) -> Dict[str, Callable]:
		for key in self.allDims:
			assert key in dimGetter, "Key '%s' is not in allDims: %s" % (key, list(dimGetter.keys()))
		return dimGetter

	def sanitizeDimTransform(self, dimTransform:Dict[str, Dict[str, Callable]]):
		for key in dimTransform:
			assert key in self.dataBuckets, "Key '%s' not in data buckets: %s" % (key, self.dataBuckets)

		for dataBucket in self.dataBuckets:
			if not dataBucket in dimTransform:
				print("[DatasetReader::sanitizeDimTransform] Data bucket '%s' not present in dimTransforms" % \
					(dataBucket))
				dimTransform[dataBucket] = {}

			for dim in self.dataBuckets[dataBucket]:
				if not dim in dimTransform[dataBucket]:
					print((("[DatasetReader::sanitizeDimTransform] Dim '%s'=>'%s' not present in ") + \
						("dimTransforms. Adding identity.")) % (dataBucket, dim))
					dimTransform[dataBucket][dim] = lambda x:x
		return dimTransform

class DatasetReader(ABC):
	def __init__(self, dataBuckets:Dict[str, List[str]], dimGetter:Dict[str, DimGetterCallable], \
		dimTransform:Dict[str, Dict[str, DimTransformCallable]]):
		self.datasetFormat = DatasetFormat(dataBuckets, dimGetter, dimTransform)
		self.dataset = None

	@abstractmethod
	def getDataset(self) -> Any:
		pass

	@abstractmethod
	def getNumData(self) -> int:
		pass

	@abstractmethod
	def getIndex(self, i:int) -> DatasetIndex:
		pass

	# Public interface

	# @brief Returns the item at index i. Basically g(i) -> Item(i). Item(i) will follow dataBuckets schema,
	#  and will call dimGetter for each dimension for this index.
	# @return The item at index i
	def getItem(self, i:int) -> Item:
		index = self.getIndex(i)
		dataBuckets = self.datasetFormat.dataBuckets
		allDims = self.datasetFormat.allDims
		dimGetter = self.datasetFormat.dimGetter
		dimTransforms = self.datasetFormat.dimTransform
		dataset = self.getDataset()
		dimToDataBuckets = self.datasetFormat.dimToDataBuckets

		result = {k : {k2 : None for k2 in dataBuckets[k]} for k in dataBuckets}
		# rawItems = {k : None for k in allDims}
		for dim in allDims:
			getterFn = dimGetter[dim]
			rawItem = getterFn(dataset, index)
			# rawItems[dim] = rawItem
			for bucket in dimToDataBuckets[dim]:
				transformFn = dimTransforms[bucket][dim]
				item = transformFn(deepcopy(rawItem))
				result[bucket][dim] = item
		return result

	# @brief The main iterator of a dataset. It will run over the data for one logical epoch.
	def iterateOneEpoch(self) -> Iterator[Dict[str, Any]]:
		n = self.getNumData()
		for i in range(n):
			index = self.getIndex(i)
			item = self.getItem(index)
			yield item

	# Generic infinite generator, that simply does a while True over the iterate_once method, which only goes one epoch
	# @param[in] type The type of processing that is generated by the generator (typicall train/test/validation)
	# @param[in] maxPrefetch How many items in advance to be generated and stored before they are consumed. If 0, the
	#  thread API is not used at all. If 1, the thread API is used with a queue of length 1 (still works better than
	#  normal in most cases, due to the multi-threaded nature. For length > 1, the queue size is just increased.
	def iterateForever(self, maxPrefetch:int=0) -> Iterator[Dict[str, np.ndarray]]:
		assert maxPrefetch >= 0
		N = self.getNumData()
		while True:
			iterateGenerator = self.iterateOneEpoch()
			if maxPrefetch > 0:
				iterateGenerator = BackgroundGenerator(iterateGenerator, max_prefetch=maxPrefetch)
			for i, items in enumerate(iterateGenerator):
				if i == N:
					break
				yield items
				del items

	def summary(self) -> str:
		return str(self)

	# # @brief Number of iterations of current epoch. May or may not change after this epoch ends.
	# def getNumIterations(self) -> int:
	# 	return len(self.getBatchSizes())

	def __str__(self) -> str:
		summaryStr = "[Dataset Reader]"
		summaryStr += "\n - Type: %s" % type(self)
		summaryStr += "\n - Data buckets:"
		for dataBucket in self.datasetFormat.dataBuckets:
			summaryStr += "\n   - %s => %s" % (dataBucket, self.datasetFormat.dataBuckets[dataBucket])
		return summaryStr

	def __len__(self) -> int:
		return self.getNumData()
