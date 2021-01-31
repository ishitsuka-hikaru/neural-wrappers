import os
from overrides import overrides
from .callback import Callback

# TODO: add format to saving files
class SaveHistory(Callback):
	def __init__(self, fileName, mode="write", **kwargs):
		self.fileName = fileName
		super().__init__(**kwargs)
		assert mode in ("write", "append")
		self.mode = "w" if mode == "write" else "a"
		self.file = None

	@overrides
	def onEpochStart(self, **kwargs):
		if self.file is None:
			self.file = open(self.fileName, mode=self.mode, buffering=1)
			self.file.write(kwargs["model"].summary() + "\n")

	@overrides
	def onEpochEnd(self, **kwargs):
		from ..pytorch.utils import _getOptimizerStr
		# SaveHistory should be just in training mode.
		if not kwargs["trainHistory"]:
			print("Warning! Using SaveHistory callback with no history (probably testing mode).")
			return

		history = kwargs["trainHistory"]
		Str = "\nEpoch %d:" % (len(history))
		Str += "\n%s" % str(history[-1])
		Str += "\n %s" % _getOptimizerStr(kwargs["model"].getOptimizer())
		self.file.write(Str)

	@overrides
	def onCallbackSave(self, **kwargs):
		if not self.file is None:
			self.file.close()
		self.file = None

	@overrides
	def onCallbackLoad(self, additional, **kwargs):
		# Make sure we're appending to the file now that we're using a loaded model (to not overwrite previous info).
		if os.path.isfile(self.fileName):
			self.file = open(self.fileName, mode="a", buffering=1)

	@overrides
	def onIterationStart(self, **kwargs):
		pass

	@overrides
	def onIterationEnd(self, results, labels, **kwargs):
		pass

	@overrides
	def __str__(self):
		return "SaveHistory (File: %s)" % (self.fileName)