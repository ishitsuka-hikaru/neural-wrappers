import numpy as np

from .video_imageio import VideoImageIO
from .video_pims import VideoPIMS

def tryReadVideo(path, vidLib="imageio", count=5, mode="array", nFrames=None, quiet=True):
	extension = path.lower().split(".")[-1]
	assert extension in ("gif", "mp4", "mov", "mkv")
	assert vidLib in ("imageio", "pims")
	assert mode in ("array", )

	f = {
		"imageio" : VideoImageIO,
		"pims" : VideoPIMS
	}[vidLib]

	i = 0
	while True:
		try:
			return f(path, nFrames)
		except Exception as e:
			print("Path: %s. Exception: %s" % (path, e))
			i += 1

			if i == count:
				raise Exception
