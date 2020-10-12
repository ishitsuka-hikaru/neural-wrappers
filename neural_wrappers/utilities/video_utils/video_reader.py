import numpy as np

def readVideoImageIo(path):
	from skimage import img_as_float32
	from skimage.color import gray2rgb
	from imageio import get_reader
	assert path.lower().endswith(".gif") or path.lower().endswith(".mp4") or path.lower().endswith(".mov")

	reader = get_reader(path)
	fps = reader.get_meta_data()["fps"]
	video = np.array([x for x in reader])

	if len(video.shape) == 3:
		video = np.array([gray2rgb(frame) for frame in video])
	if video.shape[-1] == 4:
		video = video[..., :3]
	return video, fps

def tryReadVideo(path, count=5, imgLib="imageio", mode="array", quiet=True):
	assert imgLib in ("imageio", )
	assert mode in ("array", )

	f = {
		"imageio" : readVideoImageIo
	}[imgLib]

	i = 0
	while True:
		try:
			return f(path)
		except Exception as e:
			print("Path: %s. Exception: %s" % (path, e))
			i += 1

			if i == count:
				raise Exception
