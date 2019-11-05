from .callback import Callback

# TODO: add format to saving files
class SaveHistory(Callback):
	def __init__(self, fileName, mode="write", **kwargs):
		super().__init__(**kwargs)
		assert mode in ("write", "append")
		self.mode = "w" if mode == "write" else "a"
		self.fileName = fileName
		self.file = None

	def onEpochStart(self, **kwargs):
		if self.file is None:
			self.file = open(self.fileName, mode=self.mode, buffering=1)
			self.file.write(kwargs["model"].summary() + "\n")

	def onEpochEnd(self, **kwargs):
		# SaveHistory should be just in training mode.
		if not kwargs["trainHistory"]:
			print("Warning! Using SaveHistory callback with no history (probably testing mode).")
			return

		message = kwargs["trainHistory"][-1]["message"]
		self.file.write(message + "\n")

	def onCallbackSave(self, **kwargs):
		self.file.close()
		self.file = None

	def onCallbackLoad(self, additional, **kwargs):
		# Make sure we're appending to the file now that we're using a loaded model (to not overwrite previous info).
		self.file = open(self.fileName, mode="a", buffering=1)