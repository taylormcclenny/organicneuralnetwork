

import os
import ast
import copy
from random import randrange



# SYNAPTIC_MUTATION_VALUES = [0.25, 0.125, 0.01]
SYNAPTIC_MUTATION_VALUES = [0.125, 0.06, 0.01]
NUM_CHILDREN_PER_SMV_PER_PARENT = 10                # ex. 10 children * 3 smv * 10 parents = 300 new children
NUM_OF_MUTATIONS = int(816*0.05)       # 816 Synapses from Occ. to SSNs (ex. 816*0.05 = 5%)

''' Get the Highest Generation Number '''
gen_reports_list = []
for filename in os.listdir("gen_reports/"):
    if filename.startswith('gen_'):
        gen_reports_list.append(filename)

gen_number_list = []
for gen_report in gen_reports_list:

    gen_number = gen_report.split('gen_')[1].split('_report')[0]
    if gen_number is not '':
        gen_number_list.append(int(gen_number))
    unique_gen_number_set = set(gen_number_list)
    ordered_gen_number_list = sorted(unique_gen_number_set)
    
highest_gen_number_from_gen_reports = ordered_gen_number_list.pop()

GENERATION_NUMBER = int(highest_gen_number_from_gen_reports)
NEW_GENERATION_NUMBER = GENERATION_NUMBER + 1



''' Get the Gen Report with the Highest Generation Number '''
with open(f"gen_reports/gen_{GENERATION_NUMBER}_report.json", "r") as file:
    top_10_mutations = file.read()
# Convert to Python Dictionary
top_10_mutations = ast.literal_eval(top_10_mutations)

top_10_mutations_list = []
for mutation in top_10_mutations:
    top_10_mutations_list.append(mutation)
# print(top_10_mutations_list)



''' Create Childern: Per Parent > Per Synaptic Mutation Value > # of Children >> Repeat '''

mutation_count = 1
for mutation_filename in top_10_mutations:

    ''' Load the Parent Neural Network '''
    with open(f"mutations/top_10s/{mutation_filename}", "r") as file:
        starting_post_synaptic_neighbors_dictionary = file.read()
    # Convert to Python Dictionary
    onn_map = ast.literal_eval(starting_post_synaptic_neighbors_dictionary)


    for synaptic_mutation_value in SYNAPTIC_MUTATION_VALUES:


        for _ in range(NUM_CHILDREN_PER_SMV_PER_PARENT):

            ''' Create an deep copy of the Parent NN '''
            onn_map_copy = copy.deepcopy(onn_map)

            mutation_list = []
            for _ in range(NUM_OF_MUTATIONS):
                
                neuron = str(randrange(1,66))     # random number b/t 1 & 66 (does not include 66)
                neuron = neuron.zfill(3)
                mutation_list.append(f'oc_{neuron}')                    # currently limited to Occ->Shape Specific Neurons

            # print(mutation_list)


            for neuron_to_mutate in mutation_list:

                synapse_index = randrange(12)    # random number b/t 0 & 12 (does not include 12)

                # subtract SYNAPSE_MUTATION_VALUE from synapse value
                mutated_synapse_value = round((onn_map_copy[neuron_to_mutate][1][synapse_index] - synaptic_mutation_value), 5)
                onn_map_copy[neuron_to_mutate][1][synapse_index] = mutated_synapse_value

                # print(f'MUTATION: {neuron_to_mutate}, {onn_map_copy[neuron_to_mutate][1][synapse_index]}')

            name_of_file = f'onn_map_gen_{NEW_GENERATION_NUMBER}_mutation_{mutation_count}_smv_{synaptic_mutation_value}'
            f = open(f"mutations/{name_of_file}.json", "x")
            f.write(str(onn_map_copy))
            f.close()

            onn_map_copy.clear()
            mutation_count+=1