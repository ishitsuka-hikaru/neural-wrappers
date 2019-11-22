from .node import MapNode
import numpy as np

class RGB(MapNode):
	def __init__(self):
		metrics = {
			"RGB (L1 pixel)" : RGB.RGBMetricL1Pixel,
			"RGB (L2)" : RGB.RGBMetricL2
		}
		super().__init__("RGB", numDims=3, lossFn=RGB.lossFn, metrics=metrics, groundTruthKey="rgb")

	def lossFn(y, t):
		# print("[RGB] (y)", type(y))
		# print("[RGB] (t)", type(t))
		return ((y - t)**2).mean()

	def RGBMetricL1Pixel(y, t, keyName, **k):
		# Remap y and t from [0 : 1] to [0 : 255]
		yRgbOrig = y[keyName] * 255
		tRgbOrig = t["rgb"] * 255
		return np.abs(yRgbOrig - tRgbOrig).mean()

	def RGBMetricL2(y, t, keyName, **k): 
		return ((y[keyName] - t["rgb"])**2).mean()