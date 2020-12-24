import numpy as np
import h5py
from typing import Callable, Any, Dict, List, Tuple
from returns.curry import partial
from ...h5_batched_dataset_reader import H5BatchedDatasetReader, defaultH5DimGetter
from ....utilities import tryReadNpy

class CarlaH5PathsReader(H5BatchedDatasetReader):
	def __init__(self, datasetPath:str, dataBuckets:Dict[str, List[str]], \
		desiredShape:Tuple[int, int], numNeighboursAhead:int, hyperParameters:Dict[str, Any]):
		from .normalizers import rgbNorm, depthNorm, poseNorm, opticalFlowNorm, normalNorm, \
			semanticSegmentationNorm, wireframeNorm, halftoneNorm, wireframeRegressionNorm
		from .utils import opticalFlowReader, rgbNeighbourReader, depthReadFunction, pathsReader

		rawReadFunction = tryReadNpy
		depthReadFunction = tryReadNpy

		dimGetter = {
			"rgb":partial(pathsReader, readFunction=rawReadFunction, dim="rgb"),
			"depth":partial(pathsReader, readFunction=depthReadFunction, dim="depth"),
			"pose":partial(defaultH5DimGetter, dim="position"),
			"optical_flow":partial(opticalFlowReader, readerObj=self),
			"semantic_segmentation":partial(pathsReader, readFunction=rawReadFunction, \
				dim="semantic_segmentation"),
			"wireframe":partial(pathsReader, readFunction=rawReadFunction, dim="wireframe"),
			"wireframe_regression":partial(pathsReader, readFunction=rawReadFunction, dim="wireframe"),
			"halftone":partial(pathsReader, readFunction=rawReadFunction, dim="halftone"),
			"normal":partial(pathsReader, readFunction=rawReadFunction, dim="normal"),
			"cameranormal":partial(pathsReader, readFunction=rawReadFunction, dim="cameranormal"),
			"rgbDomain2":partial(pathsReader, readFunction=rawReadFunction, dim="rgbDomain2"),
		}

		dimTransform ={
			"data":{
				"rgb":partial(rgbNorm, readerObj=self),
				"depth":partial(depthNorm, readerObj=self),
				"pose":partial(poseNorm, readerObj=self),
				"semantic_segmentation":partial(semanticSegmentationNorm, readerObj=self),
				"wireframe":partial(wireframeNorm, readerObj=self),
				"wireframe_regression":partial(wireframeRegressionNorm, readerObj=self),
				"halftone":partial(halftoneNorm, readerObj=self),
				"normal":partial(normalNorm, readerObj=self),
				"cameranormal":partial(normalNorm, readerObj=self),
				"rgbDomain2":partial(rgbNorm, readerObj=self),
			}
		}

		for i in range(abs(numNeighboursAhead)):
			ix = i * np.sign(numNeighboursAhead)
			key = "rgbNeighbour(t%+d)" % (i + 1)
			dimGetter[key] = partial(rgbNeighbourReader, skip=i + 1, readerObj=self)
			dimTransform["data"][key] = partial(rgbNorm, readerObj=self)

			flowKey = "optical_flow(t%+d)" % (i + 1)
			dimGetter[flowKey] = partial(opticalFlowReader, readerObj=self, dim=flowKey)
			dimTransform["data"][flowKey] = partial(opticalFlowNorm, readerObj=self)

		super().__init__(datasetPath, dataBuckets, dimGetter, dimTransform)
		self.rawReadFunction = rawReadFunction
		self.numNeighboursAhead = numNeighboursAhead
		self.idOfNeighbour = self.getIdsOfNeighbour()
		self.desiredShape = desiredShape
		self.hyperParameters = hyperParameters

		self.rawDepthReadFunction = lambda x : x
		self.rawFlowReadFunction = lambda x : x

	# For each top level (train/tet/val) create a new array with the index of the frame at time t + skipFrames.
	# For example result["train"][0] = 2550 means that, after randomization the frame at time=1 is at id 2550.
	def getIdsOfNeighbour(self):
		def f(ids, skipFrames):
			N = len(ids)
			closest = np.zeros(N, dtype=np.uint32)
			for i in range(N):
				where = np.where(ids == ids[i] + skipFrames)[0]
				if len(where) == 0:
					where = [i]
				assert len(where) == 1
				closest[i] = where[0]
			return closest

		result = {}
		assert "ids" in self.getDataset()
		ids = self.getDataset()["ids"][()]
		for i in range(self.numNeighboursAhead):
			key = "t+%d" % (i + 1)
			result[key] = f(ids, i + 1)
		return result

	def __getitem__(self, key):
		item = super().__getitem__(key)
		return item["data"], item["data"]

	def __len__(self) -> int:
		return len(self.getDataset()["rgb"])

	def __str__(self) -> str:
		summaryStr = "[CarlaH5PathsNpyReader]"
		summaryStr += "\n %s" % super().__str__()
		return summaryStr
