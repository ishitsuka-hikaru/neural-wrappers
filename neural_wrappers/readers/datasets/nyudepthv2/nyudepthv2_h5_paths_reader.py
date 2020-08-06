import numpy as np
import h5py
from typing import Callable, Any, Dict, List, Tuple, Optional
from functools import partial

from .normalizers import rgbNorm, depthNorm, normalNorm, semanticSegmentationNorm
from ...h5_dataset_reader import H5DatasetReader
from ...internal import DatasetIndex
from ....utilities import tryReadImage

# Append base directory to all paths read from the h5, and then call the reading function for each full path.
def pathsReader(dataset : h5py._hl.group.Group, index : DatasetIndex, readerObj : H5DatasetReader,
	readFn : Callable[[str], np.ndarray], dim : str) -> np.ndarray:
	topLevelRenames = {"semantic_segmentation" : "seg", "normal" : "normal_mask", \
		"rgb" : "img", "halftone" : "halftone", "depth" : "depth"}
	baseDirectory = readerObj.dataset["others"]["baseDirectory"][()]
	paths = dataset[dim][index.start : index.end]

	results = []
	for path in paths:
		path = "%s/%s/%s" % (baseDirectory, topLevelRenames[dim], str(path, "utf8"))
		results.append(readFn(path))
	return np.array(results)

class NYUDepthV2H5PathsReader(H5DatasetReader):
	def __init__(self, datasetPath : str, dataBuckets : Dict[str, List[str]], \
		desiredShape : Tuple[int, int], hyperParameters : Optional[Dict[str, Any]]={}, **kwargs):

		dimGetter = {
			"rgb" : partial(pathsReader, readerObj=self, readFn=tryReadImage, dim="rgb"),
			"depth" : partial(pathsReader, readerObj=self, readFn=np.load, dim="depth"),
			"semantic_segmentation" : partial(pathsReader, readerObj=self, readFn=tryReadImage,	
				dim="semantic_segmentation"),
			"halftone" : partial(pathsReader, readerObj=self, dim="halftone", readFn=tryReadImage),
			"normal" : partial(pathsReader, readerObj=self, dim="normal", readFn=tryReadImage),
		}

		dimTransform = {
			"data" : {
				"rgb" : partial(rgbNorm, readerObj=self),
				"depth" : partial(depthNorm, readerObj=self),
				"semantic_segmentation" : partial(semanticSegmentationNorm, readerObj=self),
				"halftone" : partial(rgbNorm, readerObj=self),
				"normal" : partial(normalNorm, readerObj=self),
			}
		}

		super().__init__(datasetPath, dataBuckets, dimGetter, dimTransform)
		self.desiredShape = desiredShape
		self.hyperParameters = hyperParameters

	def __str__(self) -> str:
		return "[CarlaH5PathsNpyReader] H5 File: %s" % (self.datasetPath)