import numpy as np
import h5py
from typing import Callable, Any, Dict, List, Tuple
from copy import copy
from returns.curry import partial
from ...batched_dataset_reader import H5BatchedDatasetReader
from ...batched_dataset_reader.h5_batched_dataset_reader import defaultH5DimGetter
from ....utilities import tryReadNpy, tryStr

# Append base directory to all paths read from the h5, and then call the reading function for each full path.
def pathsReader(dataset:h5py._hl.group.Group, index:np.ndarray, \
	readFunction:Callable[[str], np.ndarray], dim:str) -> np.ndarray:
	baseDirectory = dataset.file["others"]["baseDirectory"][()]
	try:
		baseDirectory = str(baseDirectory, "utf8")
	except Exception:
		pass

	try:
		paths = dataset[dim][index]
	except Exception:
		paths = smartIndexWrapper(dataset[dim], index)

	results = []
	for path in paths:
		path = "%s/%s" % (baseDirectory, str(path, "utf8"))
		results.append(readFunction(path))
	return np.array(results)


def prevReader(dataset:h5py._hl.group.Group, index, dimGetter, neighbours, delta) -> np.ndarray:
	assert delta != 0
	Key = "t%+d" % delta
	prevIndex = neighbours[Key][index]
	return dimGetter(dataset, prevIndex)

def opticalFlowReader(dataset:h5py._hl.group.Group, index, neighbours, delta:int) -> np.ndarray:
	baseDirectory = tryStr(dataset.file["others"]["baseDirectory"][()])
	assert delta != 0
	Key = "t%+d" % delta
	flowKey = "optical_flow(%s, t)" % Key

	prevIndex = neighbours[Key][index]
	paths = np.array([dataset[flowKey][ix] for ix in prevIndex])
	paths_x, paths_y = paths[:, 0], paths[:, 1]
	paths_x = ["%s/%s" % (baseDirectory, tryStr(x)) for x in paths_x]
	paths_y = ["%s/%s" % (baseDirectory, tryStr(y)) for y in paths_y]
	flow_x = np.array([tryReadNpy(x) for x in paths_x])
	flow_y = np.array([tryReadNpy(y) for y in paths_y])
	return np.stack([flow_x, flow_y], axis=-1)

class CarlaH5PathsReader(H5BatchedDatasetReader):
	def __init__(self, datasetPath:str, dataBuckets:Dict[str, List[str]], \
		deltas:List[int]=[], hyperParameters:Dict[str, Any]={}):
		from .normalizers import rgbNorm, depthNorm, poseNorm, opticalFlowNorm, normalNorm, \
			semanticSegmentationNorm, wireframeNorm, halftoneNorm, wireframeRegressionNorm

		rawReadFunction = tryReadNpy
		depthReadFunction = tryReadNpy
		flowReadFunction = opticalFlowReader
		hasOpticalFlow = False

		# Real name is optical_flow(t+k, t), so we remove the generic one because of name mismatch.
		if "optical_flow" in dataBuckets["data"]:
			hasOpticalFlow = True
			dataBuckets["data"].remove("optical_flow")

		dimGetter = {
			"rgb" : partial(pathsReader, readFunction=rawReadFunction, dim="rgb"),
			"depth" : partial(pathsReader, readFunction=depthReadFunction, dim="depth"),
			"pose" : partial(defaultH5DimGetter, dim="position"),
			"semantic_segmentation" : partial(pathsReader, readFunction=rawReadFunction, dim="semantic_segmentation"),
			"wireframe" : partial(pathsReader, readFunction=rawReadFunction, dim="wireframe"),
			"wireframe_regression" : partial(pathsReader, readFunction=rawReadFunction, dim="wireframe"),
			"halftone" : partial(pathsReader, readFunction=rawReadFunction, dim="halftone"),
			"normal" : partial(pathsReader, readFunction=rawReadFunction, dim="normal"),
			"cameranormal" : partial(pathsReader, readFunction=rawReadFunction, dim="cameranormal"),
			"rgbDomain2" : partial(pathsReader, readFunction=rawReadFunction, dim="rgbDomain2"),
		}

		dataDimTransforms = {
			"rgb" : partial(rgbNorm, readerObj=self),
			"depth" : partial(depthNorm, readerObj=self),
			"pose" : partial(poseNorm, readerObj=self),
			"semantic_segmentation" : partial(semanticSegmentationNorm, readerObj=self),
			"wireframe" : partial(wireframeNorm, readerObj=self),
			"wireframe_regression" : partial(wireframeRegressionNorm, readerObj=self),
			"halftone" : partial(halftoneNorm, readerObj=self),
			"normal" : partial(normalNorm, readerObj=self),
			"cameranormal" : partial(normalNorm, readerObj=self),
			"rgbDomain2" : partial(rgbNorm, readerObj=self),
		}
		dimTransform = {"data" : {k : dataDimTransforms[k] for k in dataBuckets["data"]}}

		# TODO: Make this more generic for use cases, not just (t-1 -> t)
		ids = datasetPath["ids"][()]
		neighbours = {"t%+d" % delta : self.getIdAtTimeDelta(ids, delta=delta) for delta in deltas}
		Keys = copy(dataBuckets["data"])

		for delta in deltas:
			if hasOpticalFlow:
				# optical_flow(t-1, t) for delta == -1
				flowKey = "optical_flow(t%+d, t)" % delta
				dimGetter[flowKey] = partial(flowReadFunction, neighbours=neighbours, delta=delta)
				dimTransform["data"][flowKey] = partial(opticalFlowNorm, readerObj=self)
				dataBuckets["data"].append(flowKey)

			# Add pose regardless for depth warping (if needed)
			poseKey = "pose(t%+d)" % delta
			dimGetter[poseKey] = partial(prevReader, dimGetter=dimGetter["pose"], neighbours=neighbours, delta=delta)
			dimTransform["data"][poseKey] = dimTransform["data"]["pose"]
			dataBuckets["data"].append(poseKey)
			# Add all (t-1) items
			for key in Keys:
				# dimKey == "rgb(t-1)" for key == "rgb" and delta == -1
				dimKey = "%s(t%+d)" % (key, delta)
				dimGetter[dimKey] = partial(prevReader, dimGetter=dimGetter[key], neighbours=neighbours, delta=delta)
				dimTransform["data"][dimKey] = dimTransform["data"][key]
				dataBuckets["data"].append(dimKey)

		super().__init__(datasetPath, dataBuckets, dimGetter, dimTransform)
		self.hyperParameters = hyperParameters
		self.desiredShape = hyperParameters["resolution"]
		self.deltas = deltas

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
		item, B = super().__getitem__(key)
		return (item["data"], item["data"]), B

	def __len__(self) -> int:
		return len(self.getDataset()["rgb"])

	def __str__(self) -> str:
		summaryStr = "[CarlaH5PathsNpyReader]"
		summaryStr += "\n %s" % super().__str__()
		return summaryStr
