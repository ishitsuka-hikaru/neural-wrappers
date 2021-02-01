import os
from pathlib import Path
from typing import Callable
from overrides import overrides
from .callback import Callback
from ..utilities import changeDirectory

_plotFn = None
_plotIterationFn = None

class RandomPlotEachEpoch(Callback):
	# @param[in] Plot function that receives the 3 arguments (x, y, t) for that particular iteration
	# @param[in] baeseDir The subdirectory where the samples are created. By default it's '$(pwd)/samples/'
	# @param[in] plotIterationFn Callback to get the iteration at which the main callback is called.
	#  By default it's 0 (start of epoch).
	def __init__(self, plotFn:Callable, baseDir:str="samples", plotIterationFn=lambda : 0):
		super().__init__(name="RandomPlotEachEpoch (Dir='%s')" % baseDir)
		self.baseDir = baseDir
		self.plotFn = plotFn
		self.plotIterationFn = plotIterationFn
		self.currentEpoch = None
		self.plotIteration = None

		global _plotFn, _plotIterationFn
		_plotFn = plotFn
		_plotIterationFn = plotIterationFn

	@overrides
	def onEpochStart(self, **kwargs):
		self.currentEpoch = kwargs["epoch"]
		self.plotIteration = self.plotIterationFn()

	@overrides
	def onIterationEnd(self, results, labels, **kwargs):
		if kwargs["iteration"] != self.plotIteration:
			return

		if kwargs["isTraining"]:
			if kwargs["isOptimizing"]:
				Dir = "%s/%d/train" % (self.baseDir, self.currentEpoch)
			else:
				Dir = "%s/%d/validation" % (self.baseDir, self.currentEpoch)
		else:
			Dir = "%s/test" % self.baseDir
		Path(Dir).mkdir(exist_ok=True, parents=True)

		cwd = os.path.realpath(os.path.abspath(os.curdir))
		os.chdir(Dir)
		self.plotFn(kwargs["data"], results, labels)
		os.chdir(cwd)

	def onCallbackLoad(self, additional, **kwargs):
		global _plotFn, _plotIterationFn
		self.plotFn = _plotFn
		self.plotIterationFn = _plotIterationFn

	def onCallbackSave(self, **kwargs):
		self.plotFn = None
		self.plotIterationFn = None
