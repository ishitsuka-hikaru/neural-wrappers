import h5py
import numpy as np
from typing import Dict, Callable, List
from functools import partial
from ...internal import DatasetIndex
from ...h5_dataset_reader import H5DatasetReader, defaultH5DimGetter
from ....utilities import tryReadImage

from .normalizers import rgbNorm, depthNorm, positionNorm
from .utils import unrealFloatFronPng

class CarlaH5PathsReader(H5DatasetReader):
	def __init__(self, datasetPath : str):#), dataBuckets : Dict[str, List[str]], \
		#dimTransform : Dict[str, Dict[str, Callable]]):
		# dimGetter : Dict[str, DimGetterCallable], 

		dataBuckets = {
			"data" : ["rgb", "depth", "position"]
		}

		dimGetter = {
			"rgb" : partial(pathsReader, readerObj=self, readFunction=rgbReader, dim="rgb"),
			"depth" : partial(pathsReader, readerObj=self, readFunction=depthReader, dim="depth"),
			"position" : partial(defaultH5DimGetter, dim="position")
		}

		dimTransform ={
			"data" : {
				"rgb" : rgbNorm,
				"depth" : partial(depthNorm, readerObj=self),
				"position" : partial(positionNorm, readerObj=self)
			}
		}
		super().__init__(datasetPath, dataBuckets, dimGetter, dimTransform)

def rgbReader(path : str) -> np.ndarray:
	return tryReadImage(path).astype(np.uint8)

def depthReader(path : str) -> np.ndarray:
	dph = tryReadImage(path)
	dph = unrealFloatFronPng(dph) * 1000
	return np.expand_dims(dph, axis=-1)

def pathsReader(dataset : h5py._hl.group.Group, index : DatasetIndex, readerObj : H5DatasetReader,
	readFunction : Callable[[str], np.ndarray], dim : str) -> np.ndarray:
	baseDirectory = readerObj.dataset["others"]["baseDirectory"][()]
	paths = dataset[dim][index.start : index.end]

	results = []
	for path in paths:
		path = "%s/%s" % (baseDirectory, str(path, "utf8"))
		results.append(readFunction(path))
	return np.array(results)