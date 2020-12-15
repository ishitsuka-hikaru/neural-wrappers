import numpy as np
from typing import Dict, List, Tuple, Any
from .carla_generic_reader import CarlaGenericReader
from .utils import unrealFloatFromPng
from ...internal import DatasetIndex
from ...batched_reader.h5_dataset_reader import H5DatasetReader, defaultH5DimGetter
from media_processing_lib.image import tryReadImage

class CarlaH5PathsReader(CarlaGenericReader):
	def __init__(self, datasetPath : str, dataBuckets : Dict[str, List[str]], \
	desiredShape : Tuple[int, int], numNeighboursAhead : int, hyperParameters : Dict[str, Any]):
		super().__init__(datasetPath, dataBuckets, tryReadImage, desiredShape, numNeighboursAhead, hyperParameters)
		self.rawDepthReadFunction = lambda path : unrealFloatFromPng(tryReadImage(path))
		self.rawFlowReadFunction = lambda path : unrealFloatFromPng(tryReadImage(path))