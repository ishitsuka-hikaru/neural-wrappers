from graphviz import Digraph

def drawGraph(nodes, edges, fileName, cleanup):
	# Convert the names into basic ones, labeled by a string digit
	dotEdges = []
	mapNodes = {}
	for i in range(len(nodes)):
		mapNodes[nodes[i]] = str(i)
	for i in range(len(edges)):
		A, B = edges[i]
		dotA, dotB = mapNodes[A], mapNodes[B]
		dotEdges.append("%s%s" % (dotA, dotB))

	# Create the graph using the original labels
	dot = Digraph(format="png")
	for node in nodes:
		dot.node(name=mapNodes[node], label=node)
	dot.edges(dotEdges)
	dot.render(fileName, view=False, cleanup=cleanup)