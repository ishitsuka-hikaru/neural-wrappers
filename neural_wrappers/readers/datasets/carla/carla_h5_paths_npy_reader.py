import numpy as np
from typing import Dict, List, Tuple, Any
from .carla_generic_reader import CarlaGenericReader
from ...internal import DatasetIndex
from ...h5_dataset_reader import H5DatasetReader, defaultH5DimGetter

def tryReadNpy(path, count=5):
	i = 0
	while True:
		try:
			return np.load(path, allow_pickle=False)
		except Exception as e:
			print("Path: %s. Exception: %s" % (path, e))
			i += 1

			if i == count:
				raise Exception

class CarlaH5PathsNpyReader(CarlaGenericReader):
	def __init__(self, datasetPath : str, dataBuckets : Dict[str, List[str]], \
	desiredShape : Tuple[int, int], numNeighboursAhead : int, hyperParameters : Dict[str, Any]):
		super().__init__(datasetPath, dataBuckets, tryReadNpy, desiredShape, numNeighboursAhead, hyperParameters)
		self.rawDepthReadFunction = tryReadNpy
		self.rawFlowReadFunction = tryReadNpy