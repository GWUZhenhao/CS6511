import igraph as ig
import numpy as np
import math
from time import time

def construct_graph(containers, target):

    select_list = []

    start_state = np.zeros(shape=len(containers))
    start_h_score = calculate_h_score(containers, start_state, target)

    g = ig.Graph() # Create the empty graph firstly

    # Create the starting node
    g.add_vertex(name = 'start',
                 state = start_state,
                 goal = (target==0),
                 g_score = 0,
                 h_score = start_h_score,
                 f_score = start_h_score + 0,
                 is_leaf = True)
    i = 0
    while check_graph(g):
        g = expand_graph(g, containers, target, select_list)
        #print(g.vs['g_score'][-1])
        '''
        if i> 15:
            break
        plot(g)
        i += 1
        '''
    return calculate_steps(g)

def calculate_steps(g):
    candidates = g.vs.select(is_leaf=True)
    nodes = candidates(f_score=np.min(candidates['f_score']))

    for node in nodes:
        if node['goal']:
            return node['g_score']
    return False

def calculate_h_score(containers, state, target):

    # if the target has been reached, h_score will be 0
    if target in state:
        return 0

    # Find the reject indies (Some containers are too small to fill the target):
    inds_reject = np.where(containers < target)

    # Find the nearest value to the target

    inds_near = np.argsort(np.absolute(state - target))
    #print('inds_near:', inds_reject, inds_near, np.absolute(state - target), state)
    ind_nearest = inds_near[0]

    for index in inds_near:
        if np.isin(index, inds_reject):
            continue
        else:
            ind_nearest = index
            break
    value_nearest = state[ind_nearest]

    # calculate the difference between target and the nearest value
    diff_target = np.absolute(value_nearest - target)
    #print('diff_target:', ind_nearest, value_nearest, target, diff_target)
    # calculate the h_score by the diff_target

    h_score = math.ceil(diff_target / np.max(containers[:-1]))*2
    #h_score = math.floor(diff_target/np.max(containers[:-1]))+1

    return h_score

def check_graph(g):
    candidates = g.vs.select(is_leaf=True)
    nodes = candidates(f_score=np.min(candidates['f_score']))

    for node in nodes:
        if node['goal']:
            return False
    #print(nodes['goal'], nodes['f_score'])
    return True

def arreq_in_list(myarr, list_arrays):
    for elem in list_arrays:
        if np.array_equal(elem, myarr):
            return True
        else:
            return False
    # return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

def expand_graph(graph, containers, target, select_list):
    # Only expanding one of leaf nodes
    candidates_to_expand = graph.vs.select(is_leaf=True)
    candidates_to_expand = candidates_to_expand(goal=False)
    node_to_expand = candidates_to_expand.find(f_score=np.min(candidates_to_expand['f_score']))
    #select_list.append(node_to_expand['state'].tolist())
    current_state = node_to_expand['state'] # Store the current state of that node.

    #
    for index, water in enumerate(current_state):

        # If the water is not full. make it full.
        if water != containers[index] and index != len(current_state) - 1:
            new_state = np.array(current_state)
            new_state[index] = containers[index]
            if new_state.tolist() not in select_list:
                graph = add_node(graph, node_to_expand, containers, target, new_state)

        # If the container has water, we can pour it to other container
        if water != 0:
            for index_target, water_target in enumerate(current_state):
                if index != index_target:
                    new_state = np.array(current_state)

                    if water <= containers[index_target] - current_state[index_target]:
                        new_state[index_target] = current_state[index_target] + water
                        new_state[index] = 0
                        if new_state.tolist() not in select_list:
                            select_list.append(new_state.tolist())
                            graph = add_node(graph, node_to_expand, containers, target, new_state)
                    else:
                        new_state[index_target] = containers[index_target]
                        new_state[index] = water - (containers[index_target] - current_state[index_target])
                        if new_state.tolist() not in select_list:
                            select_list.append(new_state.tolist())
                            graph = add_node(graph, node_to_expand, containers, target, new_state)

        # If the container has water, wa can also pour it away.
        if water != 0:
            new_state = np.array(current_state)
            new_state[index] = 0
            if new_state.tolist() not in select_list:
                select_list.append(new_state.tolist())
                graph = add_node(graph, node_to_expand, containers, target, new_state)
            
    node_to_expand['is_leaf'] = False
    return graph

def add_node(graph, node_to_expand, containers, target, new_state):

    new_h_score = calculate_h_score(containers, new_state, target)
    new_g_score = node_to_expand['g_score'] + 1
    #print(node_to_expand['state'], node_to_expand['g_score'], node_to_expand['h_score'], node_to_expand['f_score'])
    goal = target in new_state
    graph.add_vertex(state=new_state,
                     goal=goal,
                     g_score=new_g_score,
                     h_score=new_h_score,
                     f_score=new_h_score + new_g_score,
                     is_leaf=True)
    graph.add_edge(node_to_expand, graph.vs[-1], weight=1)
    return graph

def plot(graph):
    layout = graph.layout("kk")
    visual_style = {}
    visual_style["vertex_size"] = 80
    visual_style["vertex_label"] = graph.vs['state']
    visual_style["edge_width"] = graph.es['weight']
    visual_style["layout"] = layout
    visual_style["bbox"] = (2000, 2000)
    visual_style["margin"] = 200
    ig.plot(graph, **visual_style)

# containers = np.random.randint(16,25, size=[4])
# containers = np.append(containers, 999999)
# target = 20
# construct_graph(containers, target)
# print()

path = 'data.txt'
def read_file(path):
    with open(path, "r") as f:
        data = f.readlines()
        data[0] = data[0].strip('\n')
        return data

data = read_file(path)
input = data[0].split(',')
input = list(map(int, input))
target = int(data[1])

t_start=time()
gcd = np.gcd.reduce(input)
if target % gcd == 0:
    containers = np.array(input)
    containers = np.append(containers, 999999)
    num_step = construct_graph(containers, target)
    print(num_step)
else:
    print(-1)


t_end=time()
t_cost=t_end-t_start
print(t_cost)
#plot(g)




