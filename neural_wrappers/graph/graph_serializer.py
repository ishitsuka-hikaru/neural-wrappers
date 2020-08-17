from overrides import overrides
from ..pytorch.network_serializer import NetworkSerializer

class GraphSerializer(NetworkSerializer):
	@overrides
	def doSaveOptimizer(self):
		res = {}
		for edge in self.model.edges:
			res[str(edge)] = edge.serializer.doSaveOptimizer()
		return res

	@overrides
	def doLoadOptimizer(self, optimizerDict):
		for edge in self.model.edges:
			edge.serializer.doLoadOptimizer(optimizerDict[str(edge)])

	@overrides
	def doSaveCallbacks(self):
		res = {None : super().doSaveCallbacks()}
		for edge in self.model.edges:
			res[str(edge)] = edge.serializer.doSaveCallbacks()
		return res

	@overrides
	def doLoadCallbacks(self, loadedState):
		super().doLoadCallbacks(loadedState[None])
		for edge in self.model.edges:
			edge.serializer.doLoadCallbacks(loadedState[str(edge)])

	@overrides
	def doSaveHistoryDict(self):
		res = {None : super().doSaveHistoryDict()}
		for edge in self.model.edges:
			res[str(edge)] = edge.serializer.doSaveHistoryDict()
		return res

	@overrides
	def doLoadHistoryDict(self, loadedState):
		super().doLoadHistoryDict(loadedState[None])
		for edge in self.model.edges:
			edge.serializer.doLoadHistoryDict(loadedState[str(edge)])