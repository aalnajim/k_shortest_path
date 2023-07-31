import heapq
import networkx as nx
import matplotlib.pyplot as plt
import random
import time


# static variables
UPPER_WEIGHT: float = 10.0
LOWER_WEIGHT: float = 1.0

# graph parameters
n: int = 10  # the number of nodes
p: float = 0.4  # the probability of an edge being created between any two nodes

# search parameters
src: int = 0  # the source node
dest: int = 4  # the destination node
k: int = 15  # the number of shortest paths to find


def k_shortest_paths(G: nx.erdos_renyi_graph, source: int, target: int, K: int =1, weight: str ='weight', all_kshortest: bool = False):
    """Returns the the k-shortest path from source to target in a weighted graph G.

    Parameters
    ----------
    G : NetworkX Graph or DiGraph

    source : node
       Starting node

    target : node
       Ending node
       
    K : integer, optional (default=1)
       The order of the shortest path to find

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight
       For weightless graph, pass '' in.
    
    all_kshortest: boolean, optional (default=False)
        If true, returns all previous shortest path

    Returns
    -------
    lengths, paths : float, lists
       Returns a tuple with float and list.
       The float stores the length of a k-shortest path.
       The list stores the k-shortest path.  

    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.

    Examples
    --------
    >>> G=nx.complete_graph(5)    
    >>> print(k_shortest_paths(G, 0, 4, 4))
    ([1, 2, 2, 2], [[0, 4], [0, 1, 4], [0, 2, 4], [0, 3, 4]])

    Notes
    ------
    Edge weight attributes must be numerical and non-negative.
    Distances are calculated as sums of weighted edges traversed.

    """
    if source == target:
        return (0, [source]) 
    
    A: list[list[int]] = []  # A container in Yen's algorithm
    all_length: list[int] = []  # to store the delay valuses of all the paths in A
    candidate_paths: list[list[int]] = []  #  B container in Yen's algorithm
    candidate_paths_length: list[int] = []  # to store the delay values of all the paths in candidate_paths
    
    B_length: int  # the length of the current shortest path
    B: list[int]  # the current shortest path
    B_length, B = nx.single_source_dijkstra(G, source, target, weight=weight)
    
    if K ==1:
        # if only the shortest path is needed
        return B_length, B
    
    #  if more than one shortest path is needed
    A.append(B)  # add the shortest path to A
    all_length.append(B_length)  # add the delay value of the shortest path to all_length
    
    if target not in A[0]:
        # if the destination node is not reachable from the source node
        raise nx.NetworkXNoPath("node %s not reachable from %s" % (source, target))
        
    for k in range(1, K):
        
        for i in range(len(A[-1]) - 1):
            #  for all the nodes in the latest shortest path except the last node
            spur_node: int = A[-1][i]
            root_path: list[int] = A[-1][:i + 1]  # the path from the source node to the spur node (current node)
            
            if weight:
                #  if it's a weighted graph
                root_path_length = get_path_length(G, root_path, weight)  # get the path length of the root path
            
            edges_removed: list[tuple[int, int]] = []  # the list of removed edges
            if  weight:
                edge_attr: list[int] = []  # the list of removed edge weights
            
            # removes the edges from the spur node to node after it that are in the previous shortest paths
            for path in A:
                if root_path == path[:i + 1]:
                    u = path[i]
                    v = path[i + 1]
                    if (u,v) not in edges_removed:
                        if weight:
                            edge_attr.append(G[u][v][weight])
                        G.remove_edge(u, v)
                        edges_removed.append((u, v))
            

            #  this for loop to avoid cycles in the selected path
            #  for all the nodes in root path and before the spur node, remove all the edges to the other nodes
            for node in root_path[:-1]:
                for u, v, attr in list(G.edges(node, data=True)):
                    if  weight:
                        edge_attr.append(attr[weight])
                    G.remove_edge(u,v)
                    edges_removed.append((u,v))
            
            # find the shortest path after removing the edges from the spur node to the destination node
            try:
                spur_path_length: int
                spur_path: list[int] 
                spur_path_length, spur_path = nx.single_source_dijkstra(G, spur_node, target, weight=weight)  
            except:
                spur_path_length: int = 0
                spur_path: list[int] = []

            #  the total path from the source node to the destination node (the path from the source node to the spur node + the path from the spur node to the destination node)    
            total_path: list[int] = root_path[:-1] + spur_path 
            if weight:
                #  the total path delay = the delay from the source node to the spur node + the delay from the spur node to the destination node
                total_path_length: int = root_path_length + spur_path_length         
            else:
                total_path_length: int = i + spur_path_length
            if total_path_length > root_path_length and total_path not in candidate_paths and total_path not in A:
                #  adding the total path to the candidate paths if it's not in A or candidate_paths
                #  total_path_length > root_path_length is to ensure that it is not an empty path
                candidate_paths.append(total_path)
                candidate_paths_length.append(total_path_length)
                
            #  adding the removed edges back to the graph        
            for w in range(len(edges_removed)):
                u = edges_removed[w][0]
                v = edges_removed[w][1]
                G.add_edge(u,v)
                if  weight:
                    G.edges[u,v][weight]=edge_attr[w]
        
        if candidate_paths != []:
            min_weight_index = candidate_paths_length.index(min(candidate_paths_length))
            B = candidate_paths.pop(min_weight_index)
            B_length = candidate_paths_length.pop(min_weight_index)
            A.append(B)
            all_length.append(B_length)
        else:
            break
    if all_kshortest:
        return (all_length, A)
    
    return (all_length[-1], A[-1])


def get_path_length(G: nx.erdos_renyi_graph, path: list[int], weight: str = 'weight') -> int:
    ''' Helper function to calculate the length of a path in a nx graph ''' 
    length: int = 0

    if len(path) > 1:
        for i in range(len(path)-1):
            u: int = path[i]
            v: int = path[i + 1]
            
            length += G.edges[u,v][weight]
    
    return length    


def check_graph(graph: nx.erdos_renyi_graph, weights) -> None:
  ''' Helper function to check that all the nodes in the graph are reachable '''
  # Check if the graph is strongly connected
  if nx.is_strongly_connected(graph):
    print("The graph is strongly connected")
  else:
    print("The graph is not strongly connected")
    # Find the strongly connected components of the graph
    components = list(nx.strongly_connected_components(graph))
    print("The strongly connected components are:", components)
    # Add some edges between the components to make the graph strongly connected
    for i in range(len(components) - 1):
      # Pick a random node from each component
      u = random.choice(list(components[i]))
      v = random.choice(list(components[i + 1]))
      # Add an edge between them with a random weight
      weight = random.randint(1, 10)
      graph.add_edge(u, v, weight=weight)
      weights[(u, v)] = weight
    print("Added some edges to make the graph strongly connected")



def start():

  # Generate a random graph with n nodes and p edge probability
  graph: nx.erdos_renyi_graph = nx.erdos_renyi_graph(n, p, directed=True)

  # Assign random weights between LOWER_WEIGHT and UPPER_WEIGHT to the edges
  # dict[tuple(nx.erdos_renyi_graph, nx.erdos_renyi_graph), int]
  weights = {(u, v): random.randint(LOWER_WEIGHT, UPPER_WEIGHT) for (u, v) in graph.edges()}
  nx.set_edge_attributes(graph, weights, 'weight')
  check_graph(graph, weights) # making sure that the graph is strongly connected

  # Draw the graph with node labels and edge labels
  pos = nx.spring_layout(graph) # Calculate the positions of the nodes
  nx.draw(graph, pos, with_labels=True) # Draw the nodes and edges
  nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights) # Draw the edge labels
  plt.show() # Show the plot

  # Example usage of Yen's algorithm with start node src and end node dest
  for i in graph.edges():
    # print all the edges and their weights
    print(i, end=', its weight is ')
    print(f'({graph.edges[i]["weight"]}) ms')
  
  print()
  start = time.time()
  print(k_shortest_paths(graph, src, dest, k, 'weight', True))  # the implemented shortest path function
  print(f'Time taken: {time.time() - start}')
  print()

  # using the pre-built function shortest_simple_paths of networkx to create a generator of all the shortest paths  
  start = time.time()
  all_paths = nx.shortest_simple_paths(graph, src, dest, weight='weight')
  print(all_paths)
  try:
      test = []
      for i in range(k):
          test.append(((temp_path := next(all_paths)), get_path_length(graph, temp_path, 'weight')))
  except:
      pass
  print(test)
  print(f'Time taken: {time.time() - start}')

if __name__ == '__main__':
  start()