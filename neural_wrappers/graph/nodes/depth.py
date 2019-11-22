import numpy as np
from .node import MapNode
from functools import partial

class Depth(MapNode):
	def __init__(self, maxDepthMeters):
		self.maxDepthMeters = maxDepthMeters
		metrics = {
			"Depth (m)" : partial(Depth.depthMetric, maxDepthMeters=maxDepthMeters)
		}
		super().__init__("Depth", numDims=1, lossFn=Depth.lossFn, metrics=metrics, groundTruthKey="depth")

	def lossFn(y, t):
		return ((y[..., 0] - t)**2).mean()

	def depthMetric(y, t, keyName, maxDepthMeters, **k):
		# Normalize back to meters, output is in [0 : 1] representing [0 : maxDepthMeters]m
		yDepthMeters = y[keyName][..., 0] * maxDepthMeters
		tDepthMeters = t["depth"] * maxDepthMeters
		return np.abs(yDepthMeters - tDepthMeters).mean()