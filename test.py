from Utils.datasetReader import GraphReader
from Utils.utils import *

# init graph reader class
graphReader = []
graphReader.append({"GR": GraphReader(dataset="Datasets"), "title": "MY DataSets"})

# visualize metrics
for item in graphReader:
    visualize(
        params=calculateMetrics(item["GR"].get_graph()),
        G=item["GR"].get_graph(),
        title=item["title"],
    )
