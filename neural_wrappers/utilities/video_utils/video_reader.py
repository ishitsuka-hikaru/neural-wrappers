import numpy as np
from abc import ABC, abstractmethod
from overrides import overrides
from copy import copy

class NWVideo(ABC):
	def __init__(self, path, nFrames=None):
		self.path = path
		self.nFrames = nFrames

	@abstractmethod
	def copy(self):
		pass

	@abstractmethod
	def __getitem__(self, key):
		pass

	@abstractmethod
	def __len__(self):
		pass

class VideoImageIO(NWVideo):
	def __init__(self, path, nFrames=None):
		super().__init__(path, nFrames)
		self.readRaw()

	def readRaw(self):
		from skimage import img_as_float32
		from skimage.color import gray2rgb
		from imageio import get_reader
		print("[VideoImageIO] Reading raw data...")

		reader = get_reader(self.path)
		metadata = reader.get_meta_data()

		nFrames = 1<<31 if self.nFrames is None else self.nFrames
		self.fps = metadata["fps"]
		# Make this smarter
		video = []
		for i, frame in enumerate(reader):
			if i == nFrames:
				break
			video.append(frame)
		video = np.array(video)
		self.nFrames = len(video)

		if len(video.shape) == 3:
			video = np.array([gray2rgb(frame) for frame in video])
		if video.shape[-1] == 4:
			video = video[..., :3]

		self.data = video
		self.shape = video.shape

	@overrides
	def copy(self):
		item = VideoImageIO(self.path, self.nFrames)
		item.data = self.data.copy()
		item.shape = self.shape
		item.nFrames = self.nFrames
		item.fps = self.fps
		return item

	# TODO: Make this smarter
	def __getitem__(self, key):
		return self.data[key]

	def __len__(self):
		return self.nFrames

class VideoPIMS(NWVideo):
	def __init__(self, path, nFrames=None):
		super().__init__(path, nFrames)
		self.readRaw()

	def readRaw(self):
		import pims
		print("[VideoPIMS] Reading raw data...")

		video = pims.Video(self.path)
		self.fps = video.frame_rate
		self.data = video
		if self.nFrames == None:
			self.nFrames = len(video)
		self.shape = (self.nFrames, *video.frame_shape)

	@overrides
	def copy(self):
		item = VideoPIMS(self.path, self.nFrames)
		item.data = copy(self.data)
		item.shape = self.shape
		item.nFrames = self.nFrames
		item.fps = self.fps
		return item

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
