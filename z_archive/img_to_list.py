'''
import os
from PIL import Image


# print(os.getcwd())
image = Image.open("triangle_100.jpeg")

sequence_of_pixels = image.getdata()
list_of_pixels = list(sequence_of_pixels)

# print("Pixels:", len(list_of_pixels))
# print(list_of_pixels)



dictionary_of_pixels = {}
row = 1
column = 1

for pixel in list_of_pixels:

    if column > 10:
        row+=1
        column=1

    pixel = list(pixel)   # convert tuple to list
    pixel = [round(p/255, 4) for p in pixel]   # scale all values 0 to 1
    dictionary_of_pixels[f"viz_in_{row}x{column}"] = pixel

    column+=1

print(dictionary_of_pixels)
'''

### Input
# 'viz_in_1x1': [1.0, 0.9765, 0.9843]

### Intuitively has 2D Spacial Dimension & Color
# (1) '1x1'
# (2) [1.0, 0.9765, 0.9843]

### Viz > VizNeuron > NeuralNet

### Viz > VizNeuron
# (1) '1x1' > 
# (2) [1.0, 0.9765, 0.9843] > (ex. 1-10 outputs) ~[-1, 0, 1, -1, 0, 1] > signal*synapse ~[[-0.72],[0],[0.48],[-0.63],[0],[0.81]]
# 1 Viz Input to 1 VizNeuron
# VizNeuron generates excititaion-inhibition signal
# VizNeuron outputs signal via synapse, multiplying signal by post-synaptic size
# [[-0.72],[0],[0.48],[-0.63],[0],[0.81]] to [[nn_1],[nn_2],[nn_3],[nn_4],[nn_5],[nn_6]]
# [nn_1] receives signals from various Neighbors ~[[-0.72],[0.01],[-0.38],[0.23],[0.11],[0]]
# [nn_1] generates signal [-0.7499] => [-1] => signal*synapses ~[[-0.84],[0],[-0.13]] => to Neighbors
# (!) Consider:  Excitatory Neighbors & Inhibitory Neighbors
# (!) Consider:  "Prime the Path" to a "hemisphere" (excite the path to the "correct" hemisphere, inhibit vice versa)
# (!) Consider:  "Prime the Path":  [1] & [0.99] VS. [-1] & [0.99] VS. [1] & [0.01] VS. [-1] & [0.01] VS. [0]


### VizNeuron > NeuralNet
# [nn_1] generates signal [-0.7499] => [-1] => signal*synapses ~[[-0.84],[0],[-0.13]] => to Neighbors
# (!) Consider:  [nn_1] generates signal [-0.7499] => [-1] => signal*synapses ~[[-0.84],[0],[-0.13]] => to Inhibitory Neighbors [[nn_5],[nn_7],[nn_9]]
# (!) Consider:  [nn_1] generates signal [0.7499] => [1] => signal*synapses ~[[0.52],[0.86],[0]] => to Excitatory Neighbors [[nn_6],[nn_8],[nn_10]]


''' "Prime The Path" '''
''' Viz > VizNeuron > NeuralNet '''
''' 100i > 100n > 100n to 3x~33n '''

''' NeuralNet 
    General  : nn_001 to nn_100  - Moderate interconnectivity
    Circle   : nn_101 to nn_133  - High hemisphere interconnectivity, Low non-hemi interconn
    Triangle : nn_134 to nn_166  - High hemisphere interconnectivity, Low non-hemi interconn
    Square   : nn_167 to nn_200  - High hemisphere interconnectivity, Low non-hemi interconn
'''

''' Define Brain Space (1n = 1space (1x1x1)) 
    General  : nn_001 to nn_100  - [100x100x100]
    Circle   : nn_101 to nn_133  - [33x33x33]  (General-Circle = ~[133x133x133x] of space)
    Triangle : nn_134 to nn_166  - [33x33x33]  (General-Triangle = ~[133x133x133x] of space)
    Square   : nn_167 to nn_200  - [33x33x33]  (General-Square = ~[133x133x133x] of space)

    nn_001 [001x001x001] to n_100 [100x100x100]
    nn_101 [101x101x101] to nn_133 [133x133x133]
    nn_134 [134x134x134] to nn_166 [166x166x166]
    nn_167 [167x167x167] to nn_200 [200x200x200]
'''

''' "Prime the Path" Example - Triangle
    General  : nn_001 [001x001x001] to n_100 [100x100x100]  Moderately Prime-excitatory
    Circle   : nn_101 [101x101x101] to nn_133 [133x133x133]  Inhibit
    Triangle : nn_134 [134x134x134] to nn_166 [166x166x166]  Heavily Prime-excitatory
    Square   : nn_167 [167x167x167] to nn_200 [200x200x200]  Inhibit
'''

''' Neuron Connectivity
    General  : nn_001 [001x001x001] to n_100 [100x100x100]  Connect 1 to 10 General-General
    Circle   : nn_101 [101x101x101] to nn_133 [133x133x133]  Connect 1 to 10 General-Circle, Connect 1 to 10 Circle-Circle
    Triangle : nn_134 [134x134x134] to nn_166 [166x166x166]  Connect 1 to 10 General-Triangle, Connect 1 to 10 Triangle-Triangle
    Square   : nn_167 [167x167x167] to nn_200 [200x200x200]  Connect 1 to 10 General-Square, Connect 1 to 10 Square-Square
'''

''' Generate Connections
    General  : nn_001 [001x001x001] to n_100 [100x100x100]  Connect 1 to 10 General-General
    Circle   : nn_101 [101x101x101] to nn_133 [133x133x133]  Connect 1 to 10 General-Circle, Connect 1 to 10 Circle-Circle
    Triangle : nn_134 [134x134x134] to nn_166 [166x166x166]  Connect 1 to 10 General-Triangle, Connect 1 to 10 Triangle-Triangle
    Square   : nn_167 [167x167x167] to nn_200 [200x200x200]  Connect 1 to 10 General-Square, Connect 1 to 10 Square-Square   
'''

""" GENEARATE NEURONS """

# Create List of General Neurons
list_of_general_neurons = []  # nn_001 [001x001x001] to n_100 [100x100x100]
n = 1
for neuron in range(1,101):
    neuron = str(n)
    neuron = neuron.zfill(3)
    print(neuron)
    list_of_general_neurons.append(f"nn_{neuron}")
    n+=1
# print(list_of_general_neurons)

''' 
# Create List of Circle Neurons
list_of_circle_neurons = []  # nn_101 [101x101x101] to nn_133 [133x133x133] 
n = 101
for neuron in range(1,34):
    neuron = str(n)
    neuron = neuron.zfill(3)
    list_of_circle_neurons.append(f"nn_{neuron}")
    n+=1
# print(list_of_circle_neurons)

# Create List of Triangle Neurons
list_of_triangle_neurons = []  # nn_134 [134x134x134] to nn_166 [166x166x166]
n = 134
for neuron in range(1,34):
    neuron = str(n)
    neuron = neuron.zfill(3)
    list_of_triangle_neurons.append(f"nn_{neuron}")
    n+=1
# print(list_of_triangle_neurons)

# Create List of Square Neurons
list_of_square_neurons = []  # nn_167 [167x167x167] to nn_200 [200x200x200]
n = 167
for neuron in range(1,35):
    neuron = str(n)
    neuron = neuron.zfill(3)
    list_of_square_neurons.append(f"nn_{neuron}")
    n+=1
# print(list_of_square_neurons)

# Create List of General-Specific Neurons (All are in their respective Specific Hemispheres)
list_of_GSN = list_of_circle_neurons[:20] + list_of_triangle_neurons[:20] + list_of_square_neurons[:20]

'''
""" GENEARATE NEURON NEIGHBORS """
'''
import random
from random import randrange

# Create Dictionary of General-General Neuron Neighbors
dictonary_of_GGN_neighbors = {}
for neuron in list_of_general_neurons[:-20]:   # for ALL BUT last 20 Neurons
    neuron_id = int(neuron.strip('nn_'))   # get neuron id
    max_neighbor = neuron_id + 20   # set max neighbor id (~distance)
    num_of_neighbors = randrange(5, 16)   # randomly set number of neuron's neighbors b/t 5 and 15
    neighbors = []
    for num in range(num_of_neighbors):
        # TO ADD: skip if randomly generated neighbor = neuron (ie. nn_087 = nn_087)
        neighbor = str(randrange(neuron_id+1, max_neighbor))
        neighbor = int(neighbor.zfill(3))
        neighbors.append(neighbor)
    dictonary_of_GGN_neighbors[neuron] = neighbors
# print(dictonary_of_GGN_neighbors)

# Create Dictionary of General-Specific Neuron Neighbors
dictonary_of_GSN_neighbors = {}
for neuron in list_of_general_neurons[-20:]:   # for ONLY last 20 Neurons
    neuron_id = int(neuron.strip('nn_'))   # get neuron id
    num_of_neighbors = randrange(15, 46)   # randomly set number of neuron's neighbors b/t 15 and 45
    neighbors = []
    for num in range(num_of_neighbors):
        neighbor = str(random.choice(list_of_GSN))
        neighbor = int(neighbor.strip('nn_'))
        neighbors.append(neighbor)
    dictonary_of_GSN_neighbors[neuron] = neighbors
# print(dictonary_of_GSN_neighbors)
'''
'''
# Create Dictionary of Circle-Circle Neuron Neighbors
dictonary_of_CCN_neighbors = {}
for neuron in list_of_circle_neurons[:-2]:   # for ALL BUT the last two Neurons
    neuron_id = int(neuron.strip('nn_'))   # get neuron id
    max_neighbor = int(list_of_circle_neurons[-1].strip('nn_'))   # set max neighbor id (~distance) - In this S Hemisphere that's the last Neuron
    # print(max_neighbor)
    # end_cap_of_hemisphere = max_neighbor - 15
    # print("end_cap_of_hemisphere", end_cap_of_hemisphere)
    num_of_neighbors = min(randrange(5, 16), max_neighbor-neuron_id)   # randomly set number of neuron's neighbors b/t 5 and 15
    print("For Neuron:", neuron_id, "-", max_neighbor, num_of_neighbors)
    
    neighbors = []
    print(len(neighbors), num_of_neighbors)

    neighbor = str(randrange(neuron_id+1, max_neighbor))
    neighbor = int(neighbor.zfill(3))

    print(neighbor)

    # while len(neighbors) < num_of_neighbors:

    #     if neighbor not in neighbors:
    #         neighbors.append(neighbor)
    #     else:
    #         continue
        
    # dictonary_of_CCN_neighbors[neuron] = neighbors
'''




    # for num in range(num_of_neighbors):
    #     print(neighbors)
    #     neighbor = str(randrange(neuron_id+1, max_neighbor))
    #     neighbor = int(neighbor.zfill(3))
    #     if neighbor not in neighbors:
    #         neighbors.append(neighbor)

    # dictonary_of_CCN_neighbors[neuron] = neighbors

# print(dictonary_of_CCN_neighbors)



'''
# Create Dictionary of Triangle-Triangle Neuron Neighbors
dictonary_of_TTN_neighbors = {}
for neuron in list_of_triangle_neurons:
    num_of_neighbors = randrange(11)  # randomly set number of neuron's neighbors
    neighbors = []
    for num in range(num_of_neighbors):
        # TO ADD: skip if randomly generated neighbor = neuron (ie. nn_087 = nn_087)
        neighbor = str(randrange(134,167))
        neighbor = neighbor.zfill(3)
        neighbors.append(neighbor)
    dictonary_of_TTN_neighbors[neuron] = neighbors
# print(dictonary_of_TTN_neighbors)

# Create Dictionary of Square-Square Neuron Neighbors
dictonary_of_SSN_neighbors = {}
for neuron in list_of_square_neurons:
    num_of_neighbors = randrange(11)  # randomly set number of neuron's neighbors
    neighbors = []
    for num in range(num_of_neighbors):
        # TO ADD: skip if randomly generated neighbor = neuron (ie. nn_087 = nn_087)
        neighbor = str(randrange(167,201))
        neighbor = neighbor.zfill(3)
        neighbors.append(neighbor)
    dictonary_of_SSN_neighbors[neuron] = neighbors
# print(dictonary_of_SSN_neighbors)
'''

'''
NEXT GOAL:  Generate intra-hemisphere connections
NEXT GOAL:  Generate connections b/t General-Circle, General-Triangle, General-Square
THEN:  Generate synapses values for connections
THEN:  Excitatory VS. Inhibitory Neighbors
THEN:  Create "Hemisphere Activity Metric" (ie. signal propgates to "Triangle" hemisphere)
'''

''' GOAL:  Generate General, intra-hemisphere connections ('nn_001' to 'nn_080' (~800 connections))
    'nn_001' can connect to 'nn_002' through 'nn_021' (20 posibilites, random(5-15 connections), forward only)
    'nn_002' can connect to 'nn_003' through 'nn_022' (20 posibilites, random(5-15 connections), forward only)
    ...
    'nn_080' can connect to 'nn_081' through 'nn_100' (20 posibilites, random(5-15 connections), forward only)

    GOAL:  Generate connections b/t General-Circle, General-Triangle, General-Square ('nn_081' to 'nn_100' (~600 connections))
    'nn_081' can connect to Circle[:20], Triangle[:20], Square[:20], (and/or(!) General[-20:])
        (60 posibilites (3*20), random(15-45 connections), forward only)
    ...
    'nn_100' can connect to Circle[:20], Triangle[:20], Square[:20], (and/or(!) General[-20:])
        (60 posibilites, random(15-45 connections), forward only)
    
    GOAL:  Generate Specialization, intra-hemisphere connections ('nn_101' to 'nn_200' (~600-700 connections))
    'nn_101' can connect to 'nn_102' through 'nn_133' (32 posibilites, up to random(5-15 connections), forward only)
    'nn_102' can connect to 'nn_103' through 'nn_133' (31 posibilites, up to random(5-15 connections), forward only)
    ...
    'nn_132' can connect to 'nn_133' through 'nn_133' (1 posibility, up to random(5-15 connections), forward only)
'''













### Parameters
# - Neurons in Network (Viz, VizNeuron, NeuralNet)
# - Activation Value
# - Synapse-Neuron Connection
# - Synapse-Neuron Size