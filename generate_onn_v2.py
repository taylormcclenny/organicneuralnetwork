

import random
from random import randrange
import numpy as np
# random.seed(42)
# np.random.seed(42)
print("\n")

""" HELPERS """

def positive_sign(min, max):
    return float(np.round(np.random.uniform(min, max), 3))

def negative_sign(min, max):
    return float(-np.round(np.random.uniform(min, max), 3))


""" GENEARATE NEURONS """

n = 1
# Create List of Focal Cones
list_of_focal_cones = []
num_of_neurons = 9                                              # fc_001 to fc_009
for neuron in range(1,num_of_neurons+1):
    neuron = str(n)
    neuron = neuron.zfill(3)
    list_of_focal_cones.append(f"fc_{neuron}")
    n+=1
# print(list_of_focal_cones)

# Create List of Peripheral Rods
list_of_peripheral_rods = []
num_of_neurons = 12                                                 # pr_010 to pr_021
for neuron in range(1,num_of_neurons+1):
    neuron = str(n)
    neuron = neuron.zfill(3)
    list_of_peripheral_rods.append(f"pr_{neuron}")
    n+=1
# print(list_of_peripheral_rods)

''' Horizontal Cells - Consider adding or provide for evolution in training '''

n = 1
# Create List of Bipolar Neurons
list_of_bipolar_neurons = []
num_of_neurons = len(list_of_focal_cones)+len(list_of_peripheral_rods)   # bp_001 to bp_0021
for neuron in range(1,num_of_neurons+1):
    neuron = str(n)
    neuron = neuron.zfill(3)
    list_of_bipolar_neurons.append(f"bp_{neuron}")
    n+=1
# print(list_of_bipolar_neurons)

''' Amacrine Cells - Consider adding or provide for evolution in training '''

n = 1
# Create List of Ganglion Neurons
list_of_ganglion_neurons = []
num_of_neurons = len(list_of_focal_cones)+len(list_of_peripheral_rods)   # gn_001 to gn_021
for neuron in range(1,num_of_neurons+1):
    neuron = str(n)
    neuron = neuron.zfill(3)
    list_of_ganglion_neurons.append(f"gn_{neuron}")
    n+=1
# print(list_of_ganglion_neurons)

n = 1
# Create List of Occipital Neurons
list_of_occipital_neurons = []
num_of_neurons = 41                                                 # on_001 to on_041
for neuron in range(1,num_of_neurons+1):
    neuron = str(n)
    neuron = neuron.zfill(3)
    list_of_occipital_neurons.append(f"on_{neuron}")
    n+=1
# print(list_of_occipital_neurons)

'''
# Create List of Cerebellum Neurons
list_of_cerebellum_neurons = []
num_of_neurons = 32   # nn_001 to nn_025
for neuron in range(1,num_of_neurons+1):
    neuron = str(n)
    neuron = neuron.zfill(3)
    list_of_cerebellum_neurons.append(f"nn_{neuron}")
    n+=1
# print(list_of_cerebellum_neurons)
'''

n = 1
# Create List of Directions Deciding Neurons
list_of_direction_deciding_neurons = []
num_of_neurons = 9                                                  # dd_001 to dd_009
for neuron in range(1,num_of_neurons+1):
    neuron = str(n)
    neuron = neuron.zfill(3)
    list_of_direction_deciding_neurons.append(f"dd_{neuron}")
    n+=1
# print(list_of_direction_deciding_neurons)




""" GENEARATE NEURON NEIGHBORS (POST-SYNAPTIC) """


''' Vision: Rods & Cones to Bipolar Cells '''
# Set the spacial position of each Focal Cone
focal_cones = {}
focal_cones['fc_001'] = [['x4', 'y4']]      # center of focal vision
focal_cones['fc_002'] = [['x4', 'y3']]      # focal cones spiral out clockwise
focal_cones['fc_003'] = [['x5', 'y3']]
focal_cones['fc_004'] = [['x5', 'y4']]
focal_cones['fc_005'] = [['x5', 'y5']]
focal_cones['fc_006'] = [['x5', 'y4']]
focal_cones['fc_007'] = [['x5', 'y3']]
focal_cones['fc_008'] = [['x4', 'y3']]
focal_cones['fc_009'] = [['x3', 'y3']]
# print(focal_cones)

# Set the spacial position of each Peripheral Rod
peripheral_rods = {}
peripheral_rods['pr_010'] = [['x4', 'y1']]    # top of visual field (12 o'clock)
peripheral_rods['pr_011'] = [['x6', 'y1']]    # preipheral rods spiral out clockwise
peripheral_rods['pr_012'] = [['x7', 'y2']]
peripheral_rods['pr_013'] = [['x7', 'y4']]
peripheral_rods['pr_014'] = [['x7', 'y6']]
peripheral_rods['pr_015'] = [['x6', 'y7']]
peripheral_rods['pr_016'] = [['x4', 'y7']]
peripheral_rods['pr_017'] = [['x2', 'y7']]
peripheral_rods['pr_018'] = [['x1', 'y6']]
peripheral_rods['pr_019'] = [['x1', 'y4']]
peripheral_rods['pr_020'] = [['x1', 'y2']]
peripheral_rods['pr_021'] = [['x2', 'y1']]
# print(peripheral_rods)

bp_index = 0
gn_index = 0
on_index = 0

# Connect each Focal Cone to it's closest Focal Bipolar Cell & Generate Random, Positive Synapse Value
list_of_focal_bipolar_cells = []
for neuron in focal_cones:
    bp_neuron = list_of_bipolar_neurons.pop(0)          # Focal (Midget) Bipolars are removed from the list
    focal_cones[neuron].append([bp_neuron])
    focal_cones[neuron].append(
        [np.round(np.random.uniform(0.85, 0.99), 3)])   # Generate random postive synapse value close to 1.0
    list_of_focal_bipolar_cells.append(bp_neuron)       # And added to Focal Bipolar list

circular_list_of_bipolar_neurons = list_of_bipolar_neurons + list_of_bipolar_neurons

# Connect each Peripheral Rod to it's 3 closest Bipolar Cells
list_of_diffuse_bipolar_cells = []
for neuron in peripheral_rods:
    peripheral_rods[neuron].append([
        circular_list_of_bipolar_neurons[bp_index-1], 
        circular_list_of_bipolar_neurons[bp_index], 
        circular_list_of_bipolar_neurons[bp_index+1]])
    peripheral_rods[neuron].append([
        np.round(np.random.uniform(0.85, 0.99), 3),     # Generate 3 random postive synapse value close to 1.0
        np.round(np.random.uniform(0.85, 0.99), 3), 
        np.round(np.random.uniform(0.85, 0.99), 3)])
    list_of_diffuse_bipolar_cells.append(circular_list_of_bipolar_neurons[bp_index])
    bp_index+=1

''' Bipolar Cells to Ganglion Neurons '''
# Connect each Focal Bipolar Cell to it's closest Focal Ganglion Neuron
focal_bipolar_cells = {}
list_of_focal_ganglion_neurons = []
for neuron in list_of_focal_bipolar_cells:
    ganglion = list_of_ganglion_neurons.pop(0)          # Focal Ganlions are removed from the Ganglion list
    focal_bipolar_cells[neuron] = [[ganglion]]
    focal_bipolar_cells[neuron].append(                 # Generate random +/- synapse value
        [random.choice([positive_sign(0.65, 0.99),negative_sign(0.65, 0.99)])])   
    list_of_focal_ganglion_neurons.append(ganglion)     # And added to Focal Ganglion list

list_of_diffuse_ganglion_neurons = list_of_ganglion_neurons   # Remaining Ganglion Neurons are Diffuse
circular_list_of_ganglion_neurons = list_of_diffuse_ganglion_neurons + list_of_diffuse_ganglion_neurons

# Connect each Diffuse Bipolar Cell to it's 3 closest Diffuse Ganglion Neurons
diffuse_bipolar_cells = {}
for neuron in list_of_diffuse_bipolar_cells:
    diffuse_bipolar_cells[neuron] = [[
        circular_list_of_ganglion_neurons[gn_index-1], 
        circular_list_of_ganglion_neurons[gn_index], 
        circular_list_of_ganglion_neurons[gn_index+1]]]
    diffuse_bipolar_cells[neuron].append([              # Generate 3 random +/- synapse value
        random.choice([positive_sign(0.65, 0.99),negative_sign(0.65, 0.99)]),
        random.choice([positive_sign(0.65, 0.99),negative_sign(0.65, 0.99)]),
        random.choice([positive_sign(0.65, 0.99),negative_sign(0.65, 0.99)])])
    gn_index+=1

''' Ganglion Neurons to Occipital Lobe '''
# Connect each Focal Ganglion Cell to it's closest Occipital Neuron
focal_ganglion_neurons = {}
list_of_focal_occipital_neurons = []
for neuron in list_of_focal_ganglion_neurons:
    occipital_neuron = list_of_occipital_neurons.pop(0)
    focal_ganglion_neurons[neuron] = [[occipital_neuron]]
    focal_ganglion_neurons[neuron].append(              # Generate random +/- synapse value
        [random.choice([positive_sign(0.65, 0.99),negative_sign(0.65, 0.99)])])   
    list_of_focal_occipital_neurons.append(occipital_neuron)

# Create "circular" list of Occipital Neurons for connecting to Diffuse Ganglion Neurons in clockwise fashion
list_of_mid_layer_occipital_neurons = list_of_occipital_neurons[-8:]   # the last 8 in the list
list_of_outter_occipital_neurons = list_of_occipital_neurons[:-8]   # all but the last 8 in the list

circular_list_of_outter_occipital_neurons = list_of_outter_occipital_neurons + list_of_outter_occipital_neurons

# Connect each Diffuse Ganglion Cell to it's 3 closest Occipital Neurons, THEN connect the 4th
diffuse_ganglion_neurons = {}
for neuron in list_of_diffuse_ganglion_neurons:
    diffuse_ganglion_neurons[neuron] = [[
        circular_list_of_outter_occipital_neurons[on_index-1], 
        circular_list_of_outter_occipital_neurons[on_index], 
        circular_list_of_outter_occipital_neurons[on_index+1]]]
    diffuse_ganglion_neurons[neuron].append([           # Generate 3 random +/- synapse value
        random.choice([positive_sign(0.20, 0.80),negative_sign(0.20, 0.80)]),
        random.choice([positive_sign(0.20, 0.80),negative_sign(0.20, 0.80)]),
        random.choice([positive_sign(0.20, 0.80),negative_sign(0.20, 0.80)])])
    on_index+=1

diffuse_ganglion_neurons['gn_010'][0].append('on_034')  # THEN connect the 4th Occipital Neuron (mid_layer)
diffuse_ganglion_neurons['gn_011'][0].append('on_035')
diffuse_ganglion_neurons['gn_012'][0].append('on_035')
diffuse_ganglion_neurons['gn_013'][0].append('on_036')
diffuse_ganglion_neurons['gn_014'][0].append('on_037')
diffuse_ganglion_neurons['gn_015'][0].append('on_037')
diffuse_ganglion_neurons['gn_016'][0].append('on_038')
diffuse_ganglion_neurons['gn_017'][0].append('on_039')
diffuse_ganglion_neurons['gn_018'][0].append('on_039')
diffuse_ganglion_neurons['gn_019'][0].append('on_040')
diffuse_ganglion_neurons['gn_020'][0].append('on_041')
diffuse_ganglion_neurons['gn_021'][0].append('on_041')


''' Occipital Lobe to Direction Deciding Neurons '''
# Connect each Occipital Lobe "Direction Section" to it's corresponding Direction Deciding Neuron
# [9][2][3]
# [8][1][4]   # sections spiral out from center, clockwise
# [7][6][5]
section_1 = list_of_focal_occipital_neurons
section_2 = ['on_009', 'on_002', 'on_003', 'on_033', 'on_010', 'on_011', 'on_034']
section_3 = ['on_003', 'on_011', 'on_012', 'on_013', 'on_014', 'on_015', 'on_035']
section_4 = ['on_003', 'on_004', 'on_005', 'on_015', 'on_016', 'on_017', 'on_036']
section_5 = ['on_005', 'on_017', 'on_018', 'on_019', 'on_020', 'on_021', 'on_037']
section_6 = ['on_007', 'on_006', 'on_005', 'on_021', 'on_022', 'on_023', 'on_038']
section_7 = ['on_007', 'on_023', 'on_024', 'on_025', 'on_026', 'on_027', 'on_039']
section_8 = ['on_007', 'on_008', 'on_009', 'on_027', 'on_028', 'on_029', 'on_040']
section_9 = ['on_009', 'on_029', 'on_030', 'on_031', 'on_032', 'on_033', 'on_041']

occipital_lobe_neurons = {}
for neuron in list_of_focal_occipital_neurons:
    occipital_lobe_neurons[neuron] = [['dd_001']]
for neuron in list_of_occipital_neurons:
    occipital_lobe_neurons[neuron] = [[]]
for neuron in section_2:
    occipital_lobe_neurons[neuron][0].append('dd_002')
for neuron in section_3:
    occipital_lobe_neurons[neuron][0].append('dd_003')
for neuron in section_4:
    occipital_lobe_neurons[neuron][0].append('dd_004')
for neuron in section_5:
    occipital_lobe_neurons[neuron][0].append('dd_005')
for neuron in section_6:
    occipital_lobe_neurons[neuron][0].append('dd_006')
for neuron in section_7:
    occipital_lobe_neurons[neuron][0].append('dd_007')
for neuron in section_8:
    occipital_lobe_neurons[neuron][0].append('dd_008')
for neuron in section_9:
    occipital_lobe_neurons[neuron][0].append('dd_009')

for neuron in occipital_lobe_neurons:
    num_of_synapses_to_generate = len(occipital_lobe_neurons[neuron][0])
    list_of_synapses_to_generate = []
    for n in range(num_of_synapses_to_generate):
        list_of_synapses_to_generate.append(
            random.choice([positive_sign(0.20, 0.80),negative_sign(0.20, 0.80)])
        )
    occipital_lobe_neurons[neuron].append(list_of_synapses_to_generate)


''' Combine All Dictionaries '''
onn_map = {}
onn_map.update(focal_cones)
onn_map.update(peripheral_rods)
onn_map.update(focal_bipolar_cells)
onn_map.update(diffuse_bipolar_cells)
onn_map.update(focal_ganglion_neurons)
onn_map.update(diffuse_ganglion_neurons)
onn_map.update(occipital_lobe_neurons)
print(onn_map)
print("\n")



