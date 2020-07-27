from overrides import overrides
from ..pytorch.network_serializer import NetworkSerializer

class GraphSerializer(NetworkSerializer):
    @overrides
    def doSaveOptimizer(self):
        res = {}
        for edge in self.model.edges:
            res[str(edge)] = edge.serializer.doSaveOptimizer()
        return res
