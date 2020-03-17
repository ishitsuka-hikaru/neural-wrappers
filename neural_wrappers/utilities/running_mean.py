from .utils import Number, Sequence, Dict

class RunningMean:
	def __init__(self, initValue=0):
		self.value = initValue
		self.count = 0
		self.sumFn, self.getFn = self.setupFns(initValue)

	def setupFns(self, initValue):
		if type(initValue) in Number.__args__:
			return lambda a, b : a + b, lambda a, b : a / b
		elif type(initValue) in Sequence.__args__:
			return lambda a, b : a + b, lambda a, b : a / b
		elif type(initValue) in Dict.__args__:
			return lambda a, b : {k : a[k] + b[k] for k in a}, lambda a, b : {k : a[k] / (b + 1e-5) for k in a}
		assert False, "Unknown type: %s" % (type(item))

	def update(self, value, count):
		if not value is None:
			assert count > 0
			self.value = self.sumFn(self.value, value)
			self.count += count

	def get(self):
		return self.getFn(self.value, self.count)

	def __repr__(self):
		return str(self.get())

	def __str__(self):
		return str(self.get())