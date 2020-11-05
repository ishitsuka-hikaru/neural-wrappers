from abc import ABC, abstractmethod

class NWVideo(ABC):
	def __init__(self, nFrames=None):
		self.nFrames = nFrames

	@abstractmethod
	def copy(self):
		pass

	@abstractmethod
	def __getitem__(self, key):
		pass

	def __setitem__(self, key, value):
		assert False, "Cannot set values to a video object. Use video.data or video[i] to get the frame."

	@abstractmethod
	def __len__(self):
		pass