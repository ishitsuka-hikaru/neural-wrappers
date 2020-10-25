import numpy as np

class VideoImageIO:
	def __init__(self, path, nFrames=None):
		from skimage import img_as_float32
		from skimage.color import gray2rgb
		from imageio import get_reader
		reader = get_reader(path)
		metadata = reader.get_meta_data()
		if nFrames == None:
			nFrames = metadata["nframes"]
		self.fps = metadata["fps"]
		# Make this smarter
		video = []
		for i, frame in enumerate(reader):
			if i == nFrames:
				break
			video.append(frame)
		video = np.array(video)

		if len(video.shape) == 3:
			video = np.array([gray2rgb(frame) for frame in video])
		if video.shape[-1] == 4:
			video = video[..., :3]

		self.data = video
		self.shape = video.shape
		self.nFrames = nFrames

	# TODO: Make this smarter
	def __getitem__(self, key):
		return self.data[key]

	def __len__(self):
		return self.nFrames

class VideoPIMS:
	def __init__(self, path, nFrames=None):
		import pims
		video = pims.Video(path)
		self.fps = video.frame_rate
		self.data = video
		if nFrames == None:
			nFrames = len(video)
		self.shape = (nFrames, *video.frame_shape)
		self.nFrames = nFrames

	def __getitem__(self, key):
		return self.data[key]

	def __len__(self):
		return self.nFrames

def tryReadVideo(path, count=5, imgLib="imageio", mode="array", nFrames=None, quiet=True):
	assert path.lower().endswith(".gif") or path.lower().endswith(".mp4") or path.lower().endswith(".mov")
	assert imgLib in ("imageio", "pims")
	assert mode in ("array", )

	f = {
		"imageio" : VideoImageIO,
		"pims" : VideoPIMS
	}[imgLib]

	i = 0
	while True:
		try:
			return f(path, nFrames)
		except Exception as e:
			print("Path: %s. Exception: %s" % (path, e))
			i += 1

			if i == count:
				raise Exception
