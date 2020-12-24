import numpy as np
import h5py
from typing import Callable, Any, Dict, List, Tuple
from returns.curry import partial
from ...h5_batched_dataset_reader import H5BatchedDatasetReader, defaultH5DimGetter
from ....utilities import tryReadNpy

def prevReader(dataset:h5py._hl.group.Group, index, readerObj, dim) -> np.ndarray:
	prevIndex = readerObj.idOfNeighbour["t-1"][index.start : index.stop]
	return readerObj.datasetFormat.dimGetter[dim](dataset, prevIndex)

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

		# TODO: Make this more generic for use cases, not just (t-1 -> t)
		dimGetter["optical_flow(t-1, t)"] = partial(opticalFlowReader, readerObj=self, dim="optical_flow(t-1, t)")
		dimTransform["data"]["optical_flow(t-1, t)"] = partial(opticalFlowNorm, readerObj=self)
		# Add all (t-1) items
		for key in ["rgb", "depth", "wireframe_regression", "pose", "semantic_segmentation", "normal", \
			"cameranormal", "halftone"]:
			dimGetter["%s(t-1)" % key] = partial(prevReader, readerObj=self, dim=key)
			dimTransform["data"]["%s(t-1)" % key] = dimTransform["data"][key]

		super().__init__(datasetPath, dataBuckets, dimGetter, dimTransform)

		self.rawReadFunction = rawReadFunction
		self.numNeighboursAhead = numNeighboursAhead
		self.idOfNeighbour = {"t-1" : self.getIdAtTimeDelta(delta=-1)}

		self.desiredShape = desiredShape
		self.hyperParameters = hyperParameters
		self.rawDepthReadFunction = lambda x : x
		self.rawFlowReadFunction = lambda x : x

	# For each top level (train/tet/val) create a new array with the index of the frame at time t + skipFrames.
	# For example result["train"][0] = 2550 means that, after randomization the frame at time=1 is at id 2550.
	def getIdAtTimeDelta(self, delta:int):
		def f(ids, delta):
			N = len(ids)
			closest = np.zeros(N, dtype=np.uint32)
			for i in range(N):
				where = np.where(ids == ids[i] + delta)[0]
				if len(where) == 0:
					where = [i]
				assert len(where) == 1
				closest[i] = where[0]
			return closest

		result = {}
		assert "ids" in self.getDataset()
		ids = self.getDataset()["ids"][()]
		result = f(ids, delta)
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
