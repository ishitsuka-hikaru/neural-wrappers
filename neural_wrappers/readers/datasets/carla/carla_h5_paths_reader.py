import numpy as np
import h5py
from typing import Callable, Any, Dict, List, Tuple
from returns.curry import partial
from ...h5_batched_dataset_reader import H5BatchedDatasetReader, defaultH5DimGetter
from ....utilities import tryReadNpy

def prevReader(dataset:h5py._hl.group.Group, index, dimGetter, neighbours, delta) -> np.ndarray:
	assert delta == -1
	Key = "t-1"
	prevIndex = neighbours[Key][index.start : index.stop]
	return dimGetter(dataset, prevIndex)

def opticalFlowReader(dataset:h5py._hl.group.Group, index, neighbours, delta:int) -> np.ndarray:
	baseDirectory = str(dataset.file["others"]["baseDirectory"][()], "utf8")
	assert delta == -1
	Key = "t-1"
	prevIndex = neighbours[Key][index]
	flowKey = "optical_flow(%s, t)" % Key
	paths = np.array([dataset[flowKey][ix] for ix in prevIndex])
	paths_x, paths_y = paths[:, 0], paths[:, 1]
	paths_x = ["%s/%s" % (baseDirectory, str(x, "utf8")) for x in paths_x]
	paths_y = ["%s/%s" % (baseDirectory, str(y, "utf8")) for y in paths_y]
	flow_x = np.array([tryReadNpy(x) for x in paths_x])
	flow_y = np.array([tryReadNpy(y) for y in paths_y])
	return np.stack([flow_x, flow_y], axis=-1)

class CarlaH5PathsReader(H5BatchedDatasetReader):
	def __init__(self, datasetPath:str, dataBuckets:Dict[str, List[str]], \
		desiredShape:Tuple[int, int], numNeighboursAhead:int, hyperParameters:Dict[str, Any]):
		from .normalizers import rgbNorm, depthNorm, poseNorm, opticalFlowNorm, normalNorm, \
			semanticSegmentationNorm, wireframeNorm, halftoneNorm, wireframeRegressionNorm
		from .utils import pathsReader

		rawReadFunction = tryReadNpy
		depthReadFunction = tryReadNpy
		flowReadFunction = opticalFlowReader

		dimGetter = {
			"rgb":partial(pathsReader, readFunction=rawReadFunction, dim="rgb"),
			"depth":partial(pathsReader, readFunction=depthReadFunction, dim="depth"),
			"pose":partial(defaultH5DimGetter, dim="position"),
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
		ids = datasetPath["ids"][()]
		neighbours = {"t-1" : self.getIdAtTimeDelta(ids, delta=-1)}
		dimGetter["optical_flow(t-1, t)"] = partial(opticalFlowReader, neighbours=neighbours, delta=-1)
		dimTransform["data"]["optical_flow(t-1, t)"] = partial(opticalFlowNorm, readerObj=self)
		# Add all (t-1) items
		for key in ["rgb", "depth", "wireframe_regression", "pose", "semantic_segmentation", "normal", \
			"cameranormal", "halftone"]:
			dimGetter["%s(t-1)" % key] = partial(prevReader, dimGetter=dimGetter[key], \
				neighbours=neighbours, delta=-1)
			dimTransform["data"]["%s(t-1)" % key] = dimTransform["data"][key]

		super().__init__(datasetPath, dataBuckets, dimGetter, dimTransform)
		self.hyperParameters = hyperParameters
		self.desiredShape = desiredShape

		# self.rawReadFunction = rawReadFunction
		# self.numNeighboursAhead = numNeighboursAhead
		# self.idOfNeighbour = {"t-1" : self.getIdAtTimeDelta(delta=-1)}

	# For each top level (train/tet/val) create a new array with the index of the frame at time t + skipFrames.
	# For example result["train"][0] = 2550 means that, after randomization the frame at time=1 is at id 2550.
	def getIdAtTimeDelta(self, ids, delta:int):
		N = len(ids)
		closest = np.zeros(N, dtype=np.uint32)
		for i in range(N):
			where = np.where(ids == ids[i] + delta)[0]
			if len(where) == 0:
				where = [i]
			assert len(where) == 1
			closest[i] = where[0]
		return closest

	def __getitem__(self, key):
		item = super().__getitem__(key)
		return item["data"], item["data"]

	def __len__(self) -> int:
		return len(self.getDataset()["rgb"])

	def __str__(self) -> str:
		summaryStr = "[CarlaH5PathsNpyReader]"
		summaryStr += "\n %s" % super().__str__()
		return summaryStr
