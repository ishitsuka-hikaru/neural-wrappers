import numpy as np
import transforms3d.euler as txe
from typing import Dict, Union
from media_processing_lib.image import imgResize

from .carla_h5_paths_reader import CarlaH5PathsReader
from ....utilities import h5ReadDict, npGetInfo

# TODO: All norms now put data in [0 : 1]. We should look at the rederObj and if some dims want other range, transform
#  the data to that range.

def rgbNorm(x : np.ndarray, readerObj:CarlaH5PathsReader) -> np.ndarray:
	# x [MBx854x854x3] => [MBx256x256x3] :: [0 : 255]
	x = imgResize(x, height=readerObj.desiredShape[0], width=readerObj.desiredShape[1], \
		resizeLib="opencv", onlyUint8=False)
	# x :: [0 : 255] => [0: 1]
	x = x.astype(np.float32) / 255
	return x

def wireframeNorm(x : np.ndarray, readerObj:CarlaH5PathsReader) -> np.ndarray:
	# Wireframe is stored as RGB with 3 identical channels
	# x :: [0 : 255] ; (MB, H, W, 3) => (MB, H, W)
	x = x[..., 0]
	# Binarize it x :: [0 : 255] => [0 : 1]
	x = (x > 0).astype(np.uint8)
	# Resize it to desired shape
	x = imgResize(x, interpolation="nearest", height=readerObj.desiredShape[0], \
		width=readerObj.desiredShape[1], resizeLib="opencv", onlyUint8=False)
	return x.astype(np.float32)

def wireframeRegressionNorm(x : np.ndarray, readerObj : CarlaH5PathsReader) -> np.ndarray:
	return rgbNorm(x, readerObj)

def halftoneNorm(x : np.ndarray, readerObj:CarlaH5PathsReader) -> np.ndarray:
	return rgbNorm(x, readerObj)

def depthNorm(x : np.ndarray, readerObj:CarlaH5PathsReader) -> np.ndarray:
	depthStats = {"min" : 0, "max" : readerObj.hyperParameters["maxDepthMeters"]}

	x = imgResize(x, height=readerObj.desiredShape[0], width=readerObj.desiredShape[1], \
		resizeLib="opencv", onlyUint8=False)
	# Depth is stored in [0 : 1] representing up to 1000m from simulator
	x = np.clip(x * 1000, depthStats["min"], depthStats["max"])
	x = (x - depthStats["min"]) / (depthStats["max"] - depthStats["min"])
	return np.expand_dims(x, axis=-1)

def poseNorm(x : np.ndarray, readerObj:CarlaH5PathsReader) -> np.ndarray:
	def positionNorm(x : np.ndarray, readerObj:CarlaH5PathsReader) -> np.ndarray:
		if readerObj.hyperParameters["positionNormalization"] == "min_max":
			positionStats = h5ReadDict(readerObj.dataset["others"]["dataStatistics"]["position"])
			minPos, maxPos = positionStats["min"][0 : 3], positionStats["max"][0 : 3]
			x = (x - minPos) / (maxPos - minPos)
		elif readerObj.hyperParameters["positionNormalization"] == "none":
			pass

		return x

	def orientationNorm(x : np.ndarray, readerObj:CarlaH5PathsReader) -> np.ndarray:
		assert readerObj.hyperParameters["orientationRepresentation"] == "euler"

		if readerObj.hyperParameters["orientationNormalization"] == "min_max":
			# -180 : 180 => [-1 : 1]
			x = x / 180
			# [-1 : 1] => [0 : 1]
			x = (x + 1) / 2
		elif readerObj.hyperParameters["orientationNormalization"] == "none":
			pass
		elif readerObj.hyperParameters["orientationNormalization"] == "sin_cos":
			# -180 : 180 => [-1 : 1]
			x = x / 180
			# [-1 : 1] => [-pi : pi]
			x = x * np.pi
			# [-pi : pi] => [-1 : 1]
			xSin = np.sin(x)
			xCos = np.cos(x)
			x = np.concatenate([xSin, xCos], axis=-1)
		return x


	position = positionNorm(x[..., 0 : 3], readerObj)
	orientation = orientationNorm(x[..., 3 : ], readerObj)
	return np.concatenate([position, orientation], axis=-1)

def opticalFlowNorm(x : np.ndarray, readerObj:CarlaH5PathsReader) -> np.ndarray:
	# Optical flow is in [-1:1] and 100% of percentage. Result is in [0:1] using only [-x%:x%] of data.
	def opticalFlowPercentageTransform(x, opticalFlowPercentage):
		# x :: [0 : 1], centered in 0
		x = (x - 0.5) * 2
		# x :: [-1 : 1], centered in 0
		opticalFlowPercentage = np.array(opticalFlowPercentage) / 100
		flow_x = np.expand_dims(x[..., 0], axis=-1)
		flow_y = np.expand_dims(x[..., 1], axis=-1)
		# flow_x :: [-x% : x%], flow_y :: [-y% : y%]
		flow_x = np.clip(flow_x, -opticalFlowPercentage[0], opticalFlowPercentage[0])
		flow_y = np.clip(flow_y, -opticalFlowPercentage[1], opticalFlowPercentage[1])
		# flow_x in [0 : 2*x%], flow_y :: [0 : 2*y%]
		flow_x += opticalFlowPercentage[0]
		flow_y += opticalFlowPercentage[1]
		# flow_x :: [0 : 1], flow_y :: [0 : 1]
		flow_x *= 1 / (2 * opticalFlowPercentage[0])
		flow_y *= 1 / (2 * opticalFlowPercentage[1])
		# flow :: [0 : 1]
		flow = np.concatenate([flow_x, flow_y], axis=-1).astype(np.float32)
		return flow

	def opticalFlowMagnitude(x):
		# flow :: [-1 : 1] => [0 : sqrt(2)]
		x = np.hypot(x[..., 0], x[..., 1])
		# norm :: [0 : sqrt(2)] => [0 : 1]
		x = x / np.sqrt(2)
		return np.expand_dims(x, axis=-1)

	# x :: [0 : 1] => [-1 : 1]
	x = (x - 0.5) * 2
	x = imgResize(x, height=readerObj.desiredShape[0], width=readerObj.desiredShape[1], \
		resizeLib="opencv", onlyUint8=False)

	# if readerObj.hyperParameters["opticalFlowPercentage"] != (100, 100):
	# 	x = opticalFlowPercentageTransform(x, readerObj.hyperParameters["opticalFlowPercentage"])

	if readerObj.hyperParameters["opticalFlowMode"] == "xy":
		return x
	elif readerObj.hyperParameters["opticalFlowMode"] == "magnitude":
		return opticalFlowMagnitude(x)
	elif readerObj.hyperParameters["opticalFlowMode"] == "xy_plus_magnitude":
		return np.concatenate([x, opticalFlowMagnitude(x)], axis=-1)
	assert False

# def opticalFlowNorm(x : np.ndarray, readerObj:CarlaH5PathsReader) -> np.ndarray:
# 	# Data in [0 : 1]
# 	width, height = readerObj.desiredShape
# 	x = imgResize(x, height=height, width=width, resizeLib="opencv", onlyUint8=False)
# 	x[..., 0] = np.float32(np.int32(x[..., 0] * width))
# 	x[..., 1] = np.float32(np.int32(x[..., 1] * height))
# 	return x

def normalNorm(x:np.ndarray, readerObj:CarlaH5PathsReader) -> np.ndarray:
	x = imgResize(x, height=readerObj.desiredShape[0], width=readerObj.desiredShape[1], \
		resizeLib="opencv", onlyUint8=False)
	# Normals are stored as [0 - 255] on 3 channels, representing orientation of the 3 axes.
	x = x.astype(np.float32) / 255
	return x

def semanticSegmentationNorm(x:np.ndarray, readerObj:CarlaH5PathsReader) -> np.ndarray:
	labelKeys = list({
		(0, 0, 0): "Unlabeled",
		(70, 70, 70): "Building",
		(153, 153, 190): "Fence",
		(160, 170, 250): "Other",
		(60, 20, 220): "Pedestrian",
		(153, 153, 153): "Pole",
		(50, 234, 157): "Road line",
		(128, 64, 128): "Road",
		(232, 35, 244): "Sidewalk",
		(35, 142, 107): "Vegetation",
		(142, 0, 0): "Car",
		(156, 102, 102): "Wall",
		(0, 220, 220): "Traffic sign"
	}.keys())
	NC = len(labelKeys)

	if readerObj.hyperParameters["semanticNumClasses"] == NC and x.shape[-1] == NC:
		assert x.min() >= 0 and x.max() <= 1
	else:
		sumLabelKeys = list(map(lambda x : x[0] + x[1] * 256 + x[2] * 256 * 256, labelKeys))
		x = x.astype(np.uint32)
		x = x[..., 0] + x[..., 1] * 256 + x[..., 2] * 256 * 256
		for i in range(NC):
			x[x == sumLabelKeys[i]] = i
		x = x.astype(np.uint8)
		x = imgResize(x, interpolation="nearest", height=readerObj.desiredShape[0], \
			width=readerObj.desiredShape[1], resizeLib="opencv", onlyUint8=False)
		# Some fancy way of doing one-hot encoding.
		x = np.eye(NC)[x].astype(np.float32)

	return x