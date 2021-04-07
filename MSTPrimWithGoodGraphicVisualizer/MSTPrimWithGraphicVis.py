import heapq
from functools import total_ordering
import random
import matplotlib.pyplot as plt
import networkx as nx


class Graph:
    def __init__(self, list_v):
        self.graph_repr = {}

        for v in list_v:
            self.graph_repr[v] = []

    def add_edge(self, edge):
        if edge not in self.graph_repr[edge.u]:
            self.graph_repr[edge.u].append(edge)
            return 1
        return 0

    def remove_edge(self, edge):
        if edge not in self.graph_repr[edge.u]:
            return 0
        else:
            self.graph_repr[edge.u].remove(edge)
            return 1

    def get_nodes(self):
        nodes = []
        for node in self.graph_repr:
            nodes.append(node)
        return nodes

    def __iter__(self):
        return self.graph_repr.__iter__()

    def __getitem__(self, item):
        if item >= len(self.graph_repr):
            return None
        else:
            return self.graph_repr.__getitem__(item)

    def __str__(self):
        my_repr = ""
        for node in self.graph_repr:
            my_repr = my_repr + "\n"
            for edge in self.graph_repr[node]:
                my_repr = my_repr + "" + edge.__str__()

    def dfs_visit(self, curr, target, parent, color):
        color[curr] = "GRAY"
        for nei in self.graph_repr[curr]:
            if color[nei.v] == "WHITE":
                parent[nei.v] = nei
                if nei.v == target:
                    return
                else:
                    pass
                self.dfs_visit(nei.v, target, parent, color)
        color[curr] = "BLACK"

    def find_path_using_dfs(self, u, v):
        color = {}
        parent = {}

        if u == v:
            return [u]

        for node in self.graph_repr:
            color[node] = "WHITE"
            parent[node] = None

        self.dfs_visit(u, v, parent, color)

        return self.traverse_path(parent, u, v)

    @staticmethod
    def traverse_path(parent, u, v):
        path = []
        dad = parent.pop(v, None)
        while parent and (dad.u != u):
            path.append(dad)
            dad = parent.pop(dad.u, None)

        path.append(dad)
        path.reverse()
        return path


class UnDirectedGraph(Graph):

    def add_edge(self, edge):
        code = super(UnDirectedGraph, self).add_edge(edge)
        if code == 1:
            code = super(UnDirectedGraph, self).add_edge(WeightedEdge(edge.v, edge.u, edge.cost))  # Shalom my friend

        return code

    def remove_edge(self, edge):
        code = super(UnDirectedGraph, self).remove_edge(edge)
        if code == 1:
            code = super(UnDirectedGraph, self).remove_edge(WeightedEdge(edge.v, edge.u, edge.cost))

        return code

    def __str__(self):
        edges_map = []
        my_repr = ""
        for node in self.graph_repr:
            for edge in self.graph_repr[node]:
                if (WeightedEdge(edge.u, edge.v, edge.cost) not in edges_map) and (
                        WeightedEdge(edge.v, edge.u, edge.cost) not in edges_map):
                    edges_map.append(edge)
                    my_repr = my_repr + " " + edge.__str__()
        return my_repr


class Edge:
    def __init__(self, u, v):
        self.u = u
        self.v = v

    def __eq__(self, other):
        return (self.u == other.u) and (self.v == other.v)

    # def __hash__(self):
    #     return hash((self.u, self.v))

    def __str__(self):
        return f"{self.u} -> {self.v}"


@total_ordering
class WeightedEdge(Edge):
    def __init__(self, u, v, w):
        super().__init__(u, v)
        self.cost = w

    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        return "|" + super(WeightedEdge, self).__str__() + f" weight: {self.cost}" + "|"


class MstPrim:
    def __init__(self, g):
        self.prim_priority_queue = []
        self.graph = g
        self.mst_tree = UnDirectedGraph(g.get_nodes())
        self.cost = 0
        self.visited = [False] * len(g.get_nodes())

    def __str__(self):
        return self.mst_tree.__str__() + f" MST cost: {self.cost}"

    def prim(self, start_node=0):
        m = len(self.graph.get_nodes()) - 1
        edge_count = 0
        self.add_edges_to_queue(start_node)

        while self.prim_priority_queue and (edge_count != m):
            edge = heapq.heappop(self.prim_priority_queue)
            node_index = edge.v

            if self.visited[node_index]:
                continue

            if self.mst_tree.add_edge(edge) == 0:
                raise Exception("Error adding edge to MST")
            edge_count += 1
            self.cost += edge.cost

            self.add_edges_to_queue(node_index)

        if edge_count != m:
            self.cost, self.mst_tree = None, None  # no MST exists

    def get_mst_tree_graph(self):
        return self.mst_tree

    def add_edges_to_queue(self, node_index):
        self.visited[node_index] = True
        edges = self.graph[node_index]
        for edge in edges:
            if not self.visited[edge.v]:
                heapq.heappush(self.prim_priority_queue, edge)


    def add_edge_cleverly(self, edge):
        path = self.mst_tree.find_path_using_dfs(edge.u, edge.v)
        max_weight_edge = self.find_heaviest_edge(path)

        if edge.cost < max_weight_edge.cost:
            if self.mst_tree.remove_edge(max_weight_edge) == 0:
                raise Exception("Error removing edge")

            self.cost -= max_weight_edge.cost

            if self.mst_tree.add_edge(edge) == 0:
                raise Exception("Error adding edge")

            self.cost += edge.cost

            print("MST Changed")
            return True
        else:
            print("Same MST")
            return False

    @staticmethod
    def find_heaviest_edge(path):
        max_w_edge = path[0]
        for x in path:
            if x.cost > max_w_edge.cost:
                max_w_edge = x

        return max_w_edge


def visualise_graph(graph, nodes):
    vis_g = nx.Graph()

    vis_g.add_nodes_from(nodes)

    # add edges
    for node in graph:
        if node is None:
            break
        for edge in graph[node]:
            vis_g.add_edge(edge.u, edge.v, weight=edge.cost)

    pos = nx.spring_layout(vis_g)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(vis_g, pos, node_size=700)

    # labels
    nx.draw_networkx_labels(vis_g, pos, font_size=20, font_family='sans-serif')

    # edges
    nx.draw_networkx_edges(vis_g, pos, width=6)

    # weights
    labels = nx.get_edge_attributes(vis_g, 'weight')
    nx.draw_networkx_edge_labels(vis_g, pos, edge_labels=labels)

    plt.show()


def create_random_edge_in_tuple(nodes, max_weight):
    node1 = random.choice(nodes)
    node2 = random.choice(nodes)

    while node1 == node2:
        node1 = random.choice(nodes)

    return node1, node2, random.randint(1, max_weight)


def add_edges_from_tuple_list(graph, tuple_list):
    for tuple_data in tuple_list:
        graph.add_edge(WeightedEdge(tuple_data[0], tuple_data[1], tuple_data[2]))


def generate_input_for_graph(edges_n, max_weight, nodes):
    edges_tuple_list = []
    for x in range(edges_n):
        edges_tuple_list.append(create_random_edge_in_tuple(nodes, max_weight))

    return edges_tuple_list


def get_mapping_of_all_nodes_and_list_of_nodes_they_dont_have_edges_to(graph):
    smart_mapping = {}

    for node in graph:
        smart_mapping[node] = []

    for node in graph:
        for node2 in graph:
            connected = False
            for edge in graph[node2]:
                if edge.v == node or edge.u == node:
                    connected = True

            if not connected:
                smart_mapping[node].append(node2)

    return smart_mapping


def write_to_file(file_name, header, object_to_print):
    with open(file_name, 'a') as f:
        f.write(f"{header} : {str(object_to_print)} \n")


def create_edge_that_connect_not_connected_nodes(graph):
    smart_mapping = get_mapping_of_all_nodes_and_list_of_nodes_they_dont_have_edges_to(graph)

    # Pick random node
    node1 = random.choice(list(smart_mapping))
    while not smart_mapping[node1]:
        node1 = random.choice(list(smart_mapping))

    # Pick random node2 that doesn't have edge to node1
    node2 = random.choice(smart_mapping[node1])

    return WeightedEdge(node1, node2, random.randint(1, max_weight))


def get_edge_that_change_mst_and_one_that_do_not(graph):
    mst_changed = False
    first_edge_not_change_mst = True
    e = None
    edge_that_do_not_change_mst = None

    # The purpose is to test edges that connect between previously-not-connected nodes
    while not mst_changed:
        e = create_edge_that_connect_not_connected_nodes(graph)

        # Add edge, if MST didn't change - repeat
        mst_changed = mst.add_edge_cleverly(e)

        if (not mst_changed) and first_edge_not_change_mst:
            edge_that_do_not_change_mst = e
            first_edge_not_change_mst = False

    return e, edge_that_do_not_change_mst


if __name__ == '__main__':
    max_weight = 100
    nodes = list(range(20))
    edges_n = 50
    file_name = "Output.txt"

    # Clean Output file
    open(file_name, 'w').close()

    edges_tuple_list = generate_input_for_graph(edges_n, max_weight, nodes)
    write_to_file(file_name, "Input", edges_tuple_list)

    graph = UnDirectedGraph(nodes)
    add_edges_from_tuple_list(graph, edges_tuple_list)

    write_to_file(file_name, "Graph", graph)
    # visualise_graph(graph, graph.get_nodes())

    mst = MstPrim(graph)
    mst.prim()
    write_to_file(file_name, "MST", mst)
    # visualise_graph(mst.mst_tree, mst.mst_tree.get_nodes())

    edge_that_change, edge_that_dont_change = get_edge_that_change_mst_and_one_that_do_not(graph)

    write_to_file(file_name, "The edge that didn't changed the MST: ", edge_that_dont_change)
    write_to_file(file_name, "The edge that changed the MST: ", edge_that_change)

    write_to_file(file_name, "New MST: ", mst)
    # visualise_graph(mst.mst_tree, mst.mst_tree.get_nodes())

    print("DEBUG")
