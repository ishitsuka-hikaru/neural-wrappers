import numpy as np
from typing import Dict, List, Tuple, Any
from .carla_generic_reader import CarlaGenericReader
from .utils import unrealFloatFromPng
from ...internal import DatasetIndex
from ...h5_dataset_reader import H5DatasetReader, defaultH5DimGetter
from ....utilities import tryReadImage

from neural_wrappers.utilities import npGetInfo

class CarlaH5PathsReader(CarlaGenericReader):
	def __init__(self, datasetPath : str, dataBuckets : Dict[str, List[str]], \
	desiredShape : Tuple[int, int], hyperParameters : Dict[str, Any]):
		self.rawDepthReadFunction = lambda path : unrealFloatFromPng(tryReadImage(path))
		self.rawFlowReadFunction = lambda path : unrealFloatFromPng(tryReadImage(path))
		super().__init__(datasetPath, dataBuckets, tryReadImage, desiredShape, hyperParameters)