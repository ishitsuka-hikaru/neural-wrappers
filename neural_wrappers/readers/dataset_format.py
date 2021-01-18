from __future__ import annotations
from typing import Dict, List, Callable
from ..utilities import flattenList
from .dataset_types import *

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
		# Used for CachedDatasetReader. Update this if the dataset is cachable (thus immutable). This means that, we
		#  enforce the condition that self.getItem(X) will return the same Item(X) from now until the end of the
		#  universe. If this assumtpion is ever broken, the cache and the _actual_ Item(X) will be different. And we
		#  don't want that.
		self.isCacheable = False

		self.dimToDataBuckets:Dict[str, List[str]] = {dim : [] for dim in self.allDims}
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
			for dim in dimTransform[key]:
				assert dim in self.allDims, "Dim '%s' is not in allDims: %s" % (dim, self.allDims)

		for dataBucket in self.dataBuckets:
			if not dataBucket in dimTransform:
				print("[DatasetReader::sanitizeDimTransform] Data bucket '%s' not present in dimTransforms" % \
					(dataBucket))
				dimTransform[dataBucket] = {}

			for dim in self.dataBuckets[dataBucket]:
				if not dim in dimTransform[dataBucket]:
					print((("[DatasetReader::sanitizeDimTransform] Dim '%s'=>'%s' not present in ") + \
						("dimTransforms. Adding identity.")) % (dataBucket, dim))
					dimTransform[dataBucket][dim] = lambda x : x
		return dimTransform

	def __eq__(self, other:DatasetFormat) -> bool:
		return self.allDims == other.allDims and self.dataBuckets == other.dataBuckets