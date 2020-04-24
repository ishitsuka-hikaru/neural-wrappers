import numpy as np
from typing import Dict, List, Callable
from ..utilities import flattenList

class DatasetReaderV2:
	# @param[in] allDims A dictionary with all available top level names (data, label etc.) and, for each name, a list
	#  of dimensions (rgb, depth, etc.). Example: {"data" : ["rgb", "depth"], "labels" : ["depth", "semantic"]}
	# @param[in] dimGetter For each possible dimension defined above, we need to receive a method that tells us how
	#  to retrieve a batch of items. Some dimensions may be overlapped in multiple top-level names, however, they are
	#  logically the same information before transforms, so we only read it once and copy in memory if needed.
	# @param[in] dimTransform The transformations for each dimension of each top-level name. Some dimensions may
	#  overlap and if this happens we duplicate the data to ensure consistency. This may be needed for cases where
	#  the same dimension may be required in 2 formats (i.e. position as quaternions as well as unnormalized 6DoF).
	def __init__(self, allDims : Dict[str, List[str]], dimGetter : Dict[str, Callable], \
		dimTransform : Dict[str, Dict[str, Callable]]):
		self.allDims = allDims
		self.dimGetter = self.sanitizeDimGetter(dimGetter)
		self.dimTransform = dimTransform

	def sanitizeDimGetter(self, dimGetter : Dict[str, Callable]):
		allDims = list(set(flattenList(self.allDims.values())))
		for key in allDims:
			assert key in dimGetter
		return dimGetter

	def sanitizeDimTransform(self, dimTransform : Dict[str, Dict[str, Callable]]):
		allDims = list(set(flattenList(self.allDims.values())))
		allTopLevels = self.allDims.keys()
		for topLevel in allTopLevels:
			assert topLevel in dimTransform
			for dim in dimTransform[topLevel]:
				assert dim in dimTransform[topLevel]
		return dimTransform