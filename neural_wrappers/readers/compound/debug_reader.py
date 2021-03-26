from ..compound_dataset_reader import CompoundDatasetReader

class DebugReader(CompoundDatasetReader):
	def __init__(self, reader, N:int):
		self.N = N
		super().__init__(reader)

	def __len__(self):
		return self.N

	def __str__(self):
		Str = "[DebugReader]"
		Str += "\n %s" % super().__str__()
		Str += "\n - N: %d" % self.N
		return Str
