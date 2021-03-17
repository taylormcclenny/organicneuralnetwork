

import ast
import copy
from random import randrange

NUM_OF_MUTATIONS = int(816*0.05)       # 816 Synapses from Occ. to SSNs
SYNAPSE_MUTATION_VALUE = 0.01
NUM_OF_MUTANTS = 50

NEW_GEN = 2
MUTATION_COUNT = 150

# KEEP 5, 24, 48, 63, 87
mutation = 'gen_1_mutation_183_smv_0.01'



''' Load the Neural Network '''

with open(f"mutations/onn_map_{mutation}.json", "r") as file:
    starting_post_synaptic_neighbors_dictionary = file.read()
# Convert to Python Dictionary
onn_map = ast.literal_eval(starting_post_synaptic_neighbors_dictionary)


MUTATION_COUNT+=1
for _ in range(NUM_OF_MUTANTS):

    onn_map_copy = copy.deepcopy(onn_map)

    mutation_list = []
    for _ in range(NUM_OF_MUTATIONS):
        
        neuron = str(randrange(1,66))     # random number b/t 1 & 66 (does not include 66)
        neuron = neuron.zfill(3)
        mutation_list.append(f'oc_{neuron}')                    # currently limited to Occ->Shape Specific Neurons

    print(mutation_list)

    for neuron_to_mutate in mutation_list:

        synapse_index = randrange(12)    # random number b/t 0 & 12 (does not include 12)

        # subtract SYNAPSE_MUTATION_VALUE from synapse value
        mutated_synapse_value = round((onn_map_copy[neuron_to_mutate][1][synapse_index] - SYNAPSE_MUTATION_VALUE), 5)
        onn_map_copy[neuron_to_mutate][1][synapse_index] = mutated_synapse_value

        print(f'MUTATION: {neuron_to_mutate}, {onn_map_copy[neuron_to_mutate][1][synapse_index]}')

    name_of_file = f'onn_map_gen_{NEW_GEN}_mutation_{MUTATION_COUNT}_smv_{SYNAPSE_MUTATION_VALUE}'
    f = open(f"mutations/{name_of_file}.json", "x")
    f.write(str(onn_map_copy))
    f.close()

    onn_map_copy.clear()
    MUTATION_COUNT+=1