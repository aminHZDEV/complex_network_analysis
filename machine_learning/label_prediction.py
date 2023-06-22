import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation
import os
from sklearn.svm import SVC
import pandas as pd


class LabelPrediction:
    def __init__(self, G: nx.classes.graph.Graph = None):
        self._graph = G

    def predictor_lp(self, method: str = "") -> None:
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(
            [self._graph.nodes[n]["label"] for n in self._graph.nodes()]
        )
        # Convert the adjacency matrix to a dense numpy array
        adj_matrix = nx.to_numpy_array(self._graph)
        labeled_nodes = [
            n for n in self._graph.nodes() if self._graph.nodes[n]["label"] != "Unknown"
        ]
        labeled_node_labels = [self._graph.nodes[n]["label"] for n in labeled_nodes]
        labeled_nodes_encoded = label_encoder.transform(labeled_node_labels)
        node_indices = [labeled_nodes.index(n) for n in labeled_nodes]
        model = LabelPropagation(kernel=method)
        model.fit(adj_matrix[node_indices], labeled_nodes_encoded)
        unknown_nodes = [
            n for n in self._graph.nodes if self._graph.nodes[n]["label"] == "Unknown"
        ]
        node_indices = [unknown_nodes.index(n) for n in unknown_nodes]
        unknown_nodes_encoded = model.predict(adj_matrix[node_indices])
        pred_labels = label_encoder.inverse_transform(unknown_nodes_encoded)
        print(f"Method {method}")
        df = pd.read_excel(os.path.join("Datasets", "nodes.xlsx"))
        for i, node in enumerate(unknown_nodes):
            df.loc[df["NodeId"] == node, method] = pred_labels[i]
            predicted_label = pred_labels[i]
            print(f"Predicted label for node {node}: {predicted_label}")
        df.to_excel(os.path.join("Datasets", "nodes.xlsx"), index=False)

    def predictor_svm(self) -> None:
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(
            [self._graph.nodes[n]["label"] for n in self._graph.nodes()]
        )
        # Convert the adjacency matrix to a dense numpy array
        adj_matrix = nx.to_numpy_array(self._graph)
        labeled_nodes = [
            n for n in self._graph.nodes() if self._graph.nodes[n]["label"] != "Unknown"
        ]
        labeled_node_labels = [self._graph.nodes[n]["label"] for n in labeled_nodes]
        labeled_nodes_encoded = label_encoder.transform(labeled_node_labels)
        node_indices = [labeled_nodes.index(n) for n in labeled_nodes]
        model = SVC(kernel="linear")
        model.fit(adj_matrix[node_indices], labeled_nodes_encoded)
        unknown_nodes = [
            n for n in self._graph.nodes if self._graph.nodes[n]["label"] == "Unknown"
        ]
        node_indices = [unknown_nodes.index(n) for n in unknown_nodes]
        unknown_nodes_encoded = model.predict(adj_matrix[node_indices])
        pred_labels = label_encoder.inverse_transform(unknown_nodes_encoded)
        print(f"SVM")
        df = pd.read_excel(os.path.join("Datasets", "nodes.xlsx"))
        for i, node in enumerate(unknown_nodes):
            df.loc[df["NodeId"] == node, "SVM"] = pred_labels[i]
            predicted_label = pred_labels[i]
            print(f"Predicted label for node {node}: {predicted_label}")
        df.to_excel(os.path.join("Datasets", "nodes.xlsx"), index=False)
