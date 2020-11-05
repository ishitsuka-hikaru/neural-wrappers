import numpy as np
from overrides import overrides
from .nw_video import NWVideo

class VideoImageIO(NWVideo):
	def __init__(self, path, nFrames=None):
		super().__init__(nFrames)
		self.path = path
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