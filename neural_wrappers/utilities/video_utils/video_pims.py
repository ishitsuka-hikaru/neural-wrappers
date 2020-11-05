from overrides import overrides
from copy import copy
from .nw_video import NWVideo

class VideoPIMS(NWVideo):
	def __init__(self, path, nFrames=None):
		super().__init__(nFrames)
		self.path = path
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
