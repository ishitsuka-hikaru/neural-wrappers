import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any, Iterator, Union
from prefetch_generator import BackgroundGenerator
from .internal import DatasetIndex
from ..utilities import flattenList

DimGetterCallable = Union[Callable[[str, DatasetIndex], Any]]

class DatasetReader(ABC):
	# @param[in] dataBuckets A dictionary with all available data bucket names (data, label etc.) and, for each bucket,
	#  a list of dimensions (rgb, depth, etc.).
	#  Example: {"data":["rgb", "depth"], "labels":["depth", "semantic"]}
	# @param[in] dimGetter For each possible dimension defined above, we need to receive a method that tells us how
	#  to retrieve a batch of items. Some dimensions may be overlapped in multiple data bucket names, however, they are
	#  logically the same information before transforms, so we only read it once and copy in memory if needed.
	# @param[in] dimTransform The transformations for each dimension of each topdata bucket name. Some dimensions may
	#  overlap and if this happens we duplicate the data to ensure consistency. This may be needed for cases where
	#  the same dimension may be required in 2 formats (i.e. position as quaternions as well as unnormalized 6DoF).
	def __init__(self, dataBuckets:Dict[str, List[str]], dimGetter:Dict[str, DimGetterCallable], \
		dimTransform:Dict[str, Dict[str, Callable]]):
		self.dataBuckets = dataBuckets
		# allDims is a list of all dimensions, irregardless of their data bucket
		self.allDims = list(set(flattenList(self.dataBuckets.values())))
		self.dimGetter = self.sanitizeDimGetter(dimGetter)
		self.dimTransform = self.sanitizeDimTransform(dimTransform)
		self.activeTopLevel:Union[str, None] = None

	@abstractmethod
	def getDataset(self, topLevel:str) -> Any:
		raise NotImplementedError("Should have implemented this")

	# @brief Returns the number of items in a given top level name
	# @param[in] topLevel The top-level dimension that is iterated over (example: train, validation, test, etc.)
	# @return The number of items in a given top level name
	@abstractmethod
	def getNumData(self, topLevel:str) -> int:
		raise NotImplementedError("Should have implemented this")

	# @brief Returns the index object specific to this dataset for a requested batch index. This is used to logically
	#  iterate through a dataset
	# @param[in] i The index of the epoch we're trying to get dataset indexes for
	# @param[in] topLevel The top-level dimension that is iterated over (example: train, validation, test, etc.)
	# @param[in] batchSize The size of a batch that is yielded at each iteration
	# @return A DatasetIndex object with the indexes of this iteration for a specific dimension
	# @abstractmethod
	def getBatchDatasetIndex(self, i:int, topLevel:str, batchSize:int) -> DatasetIndex:
		raise NotImplementedError("Should have implemented this")

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

	def setActiveTopLevel(self, topLevel:Union[str, None]):
		self.activeTopLevel = topLevel

	def getActiveTopLevel(self) -> Union[str, None]:
		return self.activeTopLevel

	# Generic infinite generator, that simply does a while True over the iterate_once method, which only goes one epoch
	# @param[in] type The type of processing that is generated by the generator (typicall train/test/validation)
	# @param[in] miniBatchSize How many items are generated at each step
	# @param[in] maxPrefetch How many items in advance to be generated and stored before they are consumed. If 0, the
	#  thread API is not used at all. If 1, the thread API is used with a queue of length 1 (still works better than
	#  normal in most cases, due to the multi-threaded nature. For length > 1, the queue size is just increased.
	def iterate(self, topLevel:str, batchSize:int, maxPrefetch:int = 0) -> Iterator[Dict[str, np.ndarray]]:
		assert maxPrefetch >= 0
		N = self.getNumIterations(topLevel, batchSize)
		while True:
			iterateGenerator = self.iterateOneEpoch(topLevel, batchSize)
			if maxPrefetch > 0:
				iterateGenerator = BackgroundGenerator(iterateGenerator, max_prefetch=maxPrefetch)
			for i, items in enumerate(iterateGenerator):
				if i == N:
					break
				yield items
				del items

	# @brief The main iterator of a dataset. It will run over the data for one logical epoch.
	# @param[in] topLevel The top-level dimension that is iterated over (example: train, validation, test, etc.)
	# @param[in] batchSize The size of a batch that is yielded at each iteration
	# @return A generator that can be used to iterate over the dataset for one epoch
	def iterateOneEpoch(self, topLevel:str, batchSize:int) -> Iterator[Dict[str, np.ndarray]]:
		# This may be useful for some readers that want some additional information about the current top level.
		self.setActiveTopLevel(topLevel)

		dataset = self.getDataset(topLevel)
		N = self.getNumIterations(topLevel, batchSize)
		for i in range(N):
			# Get the logical index in the dataset
			index = self.getBatchDatasetIndex(i, topLevel, batchSize)
			# Get the data before bucketing, as there could be overlapping items in each bucket, just
			#  normalized differently, so we shouldn't read from disk more times.
			items, copyDims = {}, {}
			for dim in self.allDims:
				items[dim] = self.dimGetter[dim](dataset, index)
				copyDims[dim] = False

			result:Dict[str, np.ndarray] = {}
			# Go through all data buckets (data/labels etc.)
			for dataBucket in self.dataBuckets:
				result[dataBucket] = {}
				# For each bucket, go through all dims (rgb/semantic/depth etc.)
				for dim in self.dataBuckets[dataBucket]:
					item = items[dim]
					# We're making sure that if this item was in other bucket as well, it's copied so we don't alter
					#  same data memory with multiple transforms
					if copyDims[dim]:
						item = item.copy()
					copyDims[dim] = True

					# Apply this item's data transform
					item = self.dimTransform[dataBucket][dim](item)

					# Store it in this batch
					result[dataBucket][dim] = item
			yield result

		# Clear active top level as well after finishing the epoch
		self.setActiveTopLevel(None)

	# @brief Return the number of iterations in an epoch for a top level name, given a batch size.
	# @param[in] topLevel The top-level dimension that is iterated over (example: train, validation, test, etc.)
	# @param[in] batchSize The size of a batch that is yielded at each iteration
	def getNumIterations(self, topLevel:str, batchSize:int) -> int:
		N = self.getNumData(topLevel)
		return N // batchSize + (N % batchSize != 0)

	def summary(self) -> str:
		summaryStr = "[Dataset summary]\n"
		summaryStr += self.__str__() + "\n"

		summaryStr += "Data buckets:\n"
		for dataBucket in self.dataBuckets:
			summaryStr += " -  %s:%s\n" % (dataBucket, self.dataBuckets[dataBucket])
		return summaryStr

	def __str__(self) -> str:
		return "General dataset reader (%s). Update __str__ in your dataset for more details when using summary." \
			% (type(self))