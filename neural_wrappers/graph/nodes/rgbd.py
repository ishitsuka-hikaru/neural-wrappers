from .node import MapNode
from .rgb import RGB
from .depth import Depth

class RGBD(MapNode):
	def __init__(self, maxDepthMeters):
		self.maxDepthMeters = maxDepthMeters
		metrics = {
			"Depth (m)" : partial(Depth.depthMetric, maxDepthMeters=maxDepthMeters),
			"RGB (L1 pixel)" : RGB.RGBMetricL1Pixel,
			"RGB (L2)" : RGB.RGBMetricL2
		}
		super().__init__("RGBD", numDims=4, lossFn=RGBD.lossFn, metrics=metrics)

	def lossFn(y, t):
		yRgb, tRgb = y[..., 0 : 3], t[..., 0 :3]
		yDepth, tDepth = y[..., 3], t[..., 3]
		return RGB.lossFn(yRgb, tRgb) + Depth.lossFn(yDepth, tDepth)