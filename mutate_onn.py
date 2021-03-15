

import ast
import copy
from random import randrange

NUM_OF_MUTATIONS = 8
NUM_OF_MUTANTS = 9
MUTATION_COUNT = 5


''' Load the Neural Network '''
mutation = 'gen_1_mutation_5'

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
        onn_map_copy[neuron_to_mutate][1][synapse_index]-=0.05     # subtract 0.01 from synapse value
        print(f'MUTATION: {neuron_to_mutate}, {onn_map_copy[neuron_to_mutate][1][synapse_index]}')

    name_of_file = f'onn_map_gen_1_mutation_{MUTATION_COUNT}'
    f = open(f"mutations/{name_of_file}.json", "x")
    f.write(str(onn_map_copy))
    f.close()

    onn_map_copy.clear()
    MUTATION_COUNT+=1