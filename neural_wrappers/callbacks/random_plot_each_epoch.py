import os
from pathlib import Path
from typing import Callable
from overrides import overrides
from .callback import Callback
from ..utilities import changeDirectory

_plotFn = None

class RandomPlotEachEpoch(Callback):
	def __init__(self, plotFn:Callable, baseDir:str="samples"):
		super().__init__(name="RandomPlotEachEpoch (Dir='%s')" % baseDir)
		self.baseDir = baseDir
		self.plotFn = plotFn
		global _plotFn
		_plotFn = plotFn

	@overrides
	def onEpochStart(self, **kwargs):
		Dir = "%s/%d" % (self.baseDir, kwargs["epoch"])
		Path(Dir).mkdir(exist_ok=True, parents=True)
		self.currentEpoch = kwargs["epoch"]
		self.epochDir = Dir

	@overrides
	def onIterationEnd(self, results, labels, **kwargs):
		# if kwargs["iteration"] == 0 and kwargs["isOptimizing"] == False:
		if kwargs["iteration"] == 0:
			cwd = os.path.realpath(os.path.abspath(os.curdir))
			os.chdir(self.epochDir)
			self.plotFn(kwargs["data"], results, labels)
			os.chdir(cwd)

	def onCallbackLoad(self, additional, **kwargs):
		global _plotFn
		self.plotFn = _plotFn

	def onCallbackSave(self, **kwargs):
		self.plotFn = None
