
''' Signal Pass '''

import time
import ast
import concurrent.futures
import multiprocessing
import copy
import json
from random import randrange

# Measure Hemisphere Activity
def measure_activity(neurons_firing):

    for neuron in neurons_firing:
        neuron_id = int(neuron[3:])

        if 1 <= neuron_id <= 100:
            activity_dict['general_hemi']+=1
        elif 101 <= neuron_id <= 133:
            activity_dict['circle_hemi']+=1
        elif 134 <= neuron_id <= 166:
            activity_dict['triangle_hemi']+=1
        elif 167 <= neuron_id <= 199:
            activity_dict['square_hemi']+=1
        else:
            pass


# Generate Signal through 1 Neuron
def generate_signal(neuron_from_signal_dict):   # nucleus

    neuron = neuron_from_signal_dict   # neuron id
    neuron_id = neuron
    signal_input = signal_dict[neuron]   # neuron's signal

    raw_signal = sum(signal_input)

    if raw_signal > 0:
        signal = 1
        neighbors = starting_post_synaptic_neighbors_dictionary[neuron][2]
        synapse_strengths = starting_post_synaptic_neighbors_dictionary[neuron][3]
    else:
        signal = -1
        neighbors = starting_post_synaptic_neighbors_dictionary[neuron][0]
        synapse_strengths = starting_post_synaptic_neighbors_dictionary[neuron][1]

    signal_outputs = [synapse_value * signal for synapse_value in synapse_strengths]   # multiply signal by array of synapse_strengths

    # print("NEURON:", neuron)
    # print("SIGNAL:", signal)

    return signal_outputs, neighbors, neuron_id, signal


def main():
    print("\n\n\n\n\n_____ START _____\n")

    # Create Success-case Training & Failure-case Training Brain Copies
    success_post_synaptic_neighbors_dictionary = copy.deepcopy(starting_post_synaptic_neighbors_dictionary)
    failure_post_synaptic_neighbors_dictionary = copy.deepcopy(starting_post_synaptic_neighbors_dictionary)

    while signal_dict:
        print("\n\n\n_____ SIGNAL PASS - NEXT SERIES OF NEURONS FIRING _____")
        print("\nNEURONS FIRING:", list(signal_dict.keys()), "\n")

        # Update activity_dict w/ firing neurons
        measure_activity(list(signal_dict.keys()))
        
        # Run concurrent neuron firings
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(generate_signal, neuron) for neuron in signal_dict}

            new_signal_dict = {}
            update_neuron_dict = {}
            for neuron in concurrent.futures.as_completed(futures):
                synapse_outputs, neighbors, neuron_id, signal = neuron.result()

                for neighbor, synapse_output in zip(neighbors, synapse_outputs):
                    new_signal_dict.setdefault(neighbor,[]).append(synapse_output)

                update_neuron_dict[neuron_id] = signal

        ### Update Success-case Training & Failure-case Training Brain Copies
        # Review (!) - Success-case modifies ALL neuron-synapses - Devise method for neuron-synapse targetting
        # Review (should be fine) - Success-case can create multiple connections to same Neuron 
            # (ex. ['nn_031', 'nn_026', 'nn_024', 'nn_034', 'nn_025', 'nn_025'])
        for neuron in update_neuron_dict:

            # Excitatory
            if update_neuron_dict[neuron] > 0:
                
                # success case
                # print("base", success_post_synaptic_neighbors_dictionary[neuron][3])
                success_post_synaptic_neighbors_dictionary[neuron][3] = [round(x+0.01, 2) for x in success_post_synaptic_neighbors_dictionary[neuron][3]]
                # print("success", success_post_synaptic_neighbors_dictionary[neuron][3])

                # If synapse strength greater than 1, limit it to 1 and create new synapse connection.
                for synapse in success_post_synaptic_neighbors_dictionary[neuron][3]:
                    if synapse >= 1:

                        # Limit synapse to 1
                        index_of_synapse_to_set_to_1 = success_post_synaptic_neighbors_dictionary[neuron][3].index(synapse)
                        success_post_synaptic_neighbors_dictionary[neuron][3][index_of_synapse_to_set_to_1] = 1.00

                        print("\nGenerating Synapse (Excite.) from:", neuron)
                        print("__ 1+ __", success_post_synaptic_neighbors_dictionary[neuron][2])
                        neuron_int_id = int(neuron[3:])
                        
                        if 101 < neuron_int_id < 133:
                            new_synapse = str(min(neuron_int_id + randrange(1,10), 133))
                            new_synapse = new_synapse.zfill(3)
                            success_post_synaptic_neighbors_dictionary[neuron][2].append(f"nn_{new_synapse}")  # generate synapse
                            success_post_synaptic_neighbors_dictionary[neuron][3].append(0.05)  # generate low synapse strength
                            print(">>>>>>>>", success_post_synaptic_neighbors_dictionary[neuron][2])
                        elif 134 < neuron_int_id < 166:
                            new_synapse = str(min(neuron_int_id + randrange(1,10), 166))
                            new_synapse = new_synapse.zfill(3)
                            success_post_synaptic_neighbors_dictionary[neuron][2].append(f"nn_{new_synapse}")
                            success_post_synaptic_neighbors_dictionary[neuron][3].append(0.05)
                            print(">>>>>>>>", success_post_synaptic_neighbors_dictionary[neuron][2])
                        elif 167 < neuron_int_id < 199:
                            new_synapse = str(min(neuron_int_id + randrange(1,10), 199))
                            new_synapse = new_synapse.zfill(3)
                            success_post_synaptic_neighbors_dictionary[neuron][2].append(f"nn_{new_synapse}")
                            success_post_synaptic_neighbors_dictionary[neuron][3].append(0.05)
                            print(">>>>>>>>", success_post_synaptic_neighbors_dictionary[neuron][2])
                        else:
                            new_synapse = str(min(neuron_int_id + randrange(1,10), 100))
                            new_synapse = new_synapse.zfill(3)
                            success_post_synaptic_neighbors_dictionary[neuron][2].append(f"nn_{new_synapse}")
                            success_post_synaptic_neighbors_dictionary[neuron][3].append(0.05)
                            print(">>>>>>>>", success_post_synaptic_neighbors_dictionary[neuron][2])
                
                # failure case
                failure_post_synaptic_neighbors_dictionary[neuron][3] = [round(x-0.01, 2) for x in failure_post_synaptic_neighbors_dictionary[neuron][3]]
                # print("failure", failure_post_synaptic_neighbors_dictionary[neuron][3])

                # If synapse strength less than 0, remove it.
                for synapse in failure_post_synaptic_neighbors_dictionary[neuron][3]:
                    if synapse <= 0:

                        print("\nRemoving Synapse from:", neuron)
                        print("__ -0 __", failure_post_synaptic_neighbors_dictionary[neuron][2])
                        print("__ -0 __", failure_post_synaptic_neighbors_dictionary[neuron][3])
                        index_of_synapse_to_remove = failure_post_synaptic_neighbors_dictionary[neuron][3].index(synapse)
                        failure_post_synaptic_neighbors_dictionary[neuron][2].pop(index_of_synapse_to_remove)
                        failure_post_synaptic_neighbors_dictionary[neuron][3].pop(index_of_synapse_to_remove)
                        print(">>>>>>>>", failure_post_synaptic_neighbors_dictionary[neuron][2])
                        print(">>>>>>>>", failure_post_synaptic_neighbors_dictionary[neuron][3])

            # Inhibitory
            else:
                
                # success case
                # print("base", success_post_synaptic_neighbors_dictionary[neuron][1])
                success_post_synaptic_neighbors_dictionary[neuron][1] = [round(x+0.01, 2) for x in success_post_synaptic_neighbors_dictionary[neuron][1]]
                # print("success", success_post_synaptic_neighbors_dictionary[neuron][1])

                for synapse in success_post_synaptic_neighbors_dictionary[neuron][1]:
                    if synapse >= 1:

                        # Limit synapse to 1
                        index_of_synapse_to_set_to_1 = success_post_synaptic_neighbors_dictionary[neuron][1].index(synapse)
                        success_post_synaptic_neighbors_dictionary[neuron][1][index_of_synapse_to_set_to_1] = 1.00

                        print("\nGenerating Synapse (Inhib.) from:", neuron)
                        print("__ 1+ __", success_post_synaptic_neighbors_dictionary[neuron][0])
                        neuron_int_id = int(neuron[3:])
                        
                        if 101 < neuron_int_id < 133:
                            new_synapse = str(min(neuron_int_id + randrange(1,10), 133))
                            new_synapse = new_synapse.zfill(3)
                            success_post_synaptic_neighbors_dictionary[neuron][0].append(f"nn_{new_synapse}")  # generate synapse
                            success_post_synaptic_neighbors_dictionary[neuron][1].append(0.05)  # generate low synapse strength
                            print(">>>>>>>>", success_post_synaptic_neighbors_dictionary[neuron][0])
                            print(success_post_synaptic_neighbors_dictionary[neuron][1])
                        elif 134 < neuron_int_id < 166:
                            new_synapse = str(min(neuron_int_id + randrange(1,10), 166))
                            new_synapse = new_synapse.zfill(3)
                            success_post_synaptic_neighbors_dictionary[neuron][0].append(f"nn_{new_synapse}")
                            success_post_synaptic_neighbors_dictionary[neuron][1].append(0.05)
                            print(">>>>>>>>", success_post_synaptic_neighbors_dictionary[neuron][0])
                            print(success_post_synaptic_neighbors_dictionary[neuron][1])
                        elif 167 < neuron_int_id < 199:
                            new_synapse = str(min(neuron_int_id + randrange(1,10), 199))
                            new_synapse = new_synapse.zfill(3)
                            success_post_synaptic_neighbors_dictionary[neuron][0].append(f"nn_{new_synapse}")
                            success_post_synaptic_neighbors_dictionary[neuron][1].append(0.05)
                            print(">>>>>>>>", success_post_synaptic_neighbors_dictionary[neuron][0])
                            print(success_post_synaptic_neighbors_dictionary[neuron][1])
                        else:
                            new_synapse = str(min(neuron_int_id + randrange(1,10), 100))
                            new_synapse = new_synapse.zfill(3)
                            success_post_synaptic_neighbors_dictionary[neuron][0].append(f"nn_{new_synapse}")
                            success_post_synaptic_neighbors_dictionary[neuron][1].append(0.05)
                            print(">>>>>>>>", success_post_synaptic_neighbors_dictionary[neuron][0])
                            print(success_post_synaptic_neighbors_dictionary[neuron][1])
                
                # failure case
                failure_post_synaptic_neighbors_dictionary[neuron][1] = [round(x-0.01, 2) for x in failure_post_synaptic_neighbors_dictionary[neuron][1]]
                # print("failure", failure_post_synaptic_neighbors_dictionary[neuron][1])

                # If synapse strength less than 0, remove it.
                for synapse in failure_post_synaptic_neighbors_dictionary[neuron][1]:
                    if synapse <= 0:

                        print("\nRemoving Synapse from:", neuron)
                        print("__ -0 __", failure_post_synaptic_neighbors_dictionary[neuron][0])
                        print("__ -0 __", failure_post_synaptic_neighbors_dictionary[neuron][1])
                        index_of_synapse_to_remove = failure_post_synaptic_neighbors_dictionary[neuron][1].index(synapse)
                        failure_post_synaptic_neighbors_dictionary[neuron][0].pop(index_of_synapse_to_remove)
                        failure_post_synaptic_neighbors_dictionary[neuron][1].pop(index_of_synapse_to_remove)
                        print(">>>>>>>>", failure_post_synaptic_neighbors_dictionary[neuron][0])
                        print(">>>>>>>>", failure_post_synaptic_neighbors_dictionary[neuron][1])

        # Update signal_dict with which neurons to fire next
        signal_dict.clear()
        signal_dict.update(new_signal_dict)
        print("\nUPDATED SIGNAL PATH:", signal_dict)
        print("\nHEMISPHERE ACTIVITY:", activity_dict)
        time.sleep(1)
    
    # Write Success-case Brain & Failure-case Brain to file
    try: 
        success_json = json.dumps(success_post_synaptic_neighbors_dictionary)
        success_brain_file = open('success_post_synaptic_neighbors_dictionary.json', 'w') 
        success_brain_file.write(success_json) 
        success_brain_file.close() 

        failure_json = json.dumps(failure_post_synaptic_neighbors_dictionary)
        failure_brain_file = open('failure_post_synaptic_neighbors_dictionary.json', 'w') 
        failure_brain_file.write(failure_json) 
        failure_brain_file.close() 
    except: 
        print("Unable to write to file")


''' RUN '''

if __name__ == '__main__':

    activity_dict = {
        'general_hemi' : 0,
        'circle_hemi' : 0,
        'triangle_hemi' : 0,
        'square_hemi' : 0,
    }
        
    # Fake signals to mimic Pixel NN Output
    signal_dict = {   # origianl
        'gn_001': 
            [-0.37, 0.95, -0.73, 0.6, -0.16],
        'gn_002':
            [0.97, 0.83, 0.21],
        'gn_003':
            [-0.37, 0.46, -0.79, -0.2, 0.51, -0.59],
        'gn_004':
            [0.95, 0.97, 0.81, 0.3, 0.1],
        'gn_005':
            [-0.91, 0.26, -0.66, -0.31, 0.52, -0.55],
        'gn_006':
            [0.89, 0.6, 0.92, 0.09],
        'gn_007':
            [-0.83, 0.36, -0.28, 0.54, -0.14],
        'gn_008':
            [0.77, 0.2, 0.01, 0.82],
        'gn_009':
            [0.12, 0.86, 0.62, 0.33, 0.06],
    }
    
    # Sample Image (triangle)
    # Height x Width: 13px x 13px
    # Total # of Pixels: 169
    sample_pixel_matrix = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.004, 0.0, 0.012, 0.318, 0.024, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.008, 0.0, 0.173, 1.0, 0.208, 0.0, 0.012, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.004, 0.0, 0.012, 0.608, 1.0, 0.667, 0.027, 0.0, 0.004, 0.0, 0.0],
        [0.0, 0.0, 0.008, 0.0, 0.278, 1.0, 0.824, 1.0, 0.318, 0.0, 0.008, 0.0, 0.0],
        [0.0, 0.004, 0.0, 0.055, 0.678, 0.792, 0.725, 0.776, 0.733, 0.071, 0.0, 0.004, 0.0],
        [0.0, 0.0, 0.0, 0.024, 0.051, 0.043, 0.047, 0.043, 0.051, 0.027, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.004, 0.008, 0.008, 0.008, 0.004, 0.008, 0.004, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    # Standard Use Dictionary for Testing
    starting_post_synaptic_neighbors_dictionary = {   # neuron_to_post-synaptic

        'fc_001': [
            ['x4', 'y4'],
            ['bp_001'],
            [0.876]
        ],
        'fc_002': [
            ['x4', 'y3'],
            ['bp_002'],
            [0.853]
        ],
        'fc_003': [
            ['x5', 'y3'],
            ['bp_003'],
            [0.952]
        ],
        'fc_004': [
            ['x5', 'y4'],
            ['bp_004'],
            [0.869]
        ],
        'fc_005': [
            ['x5', 'y5'],
            ['bp_005'],
            [0.977]
        ],
        'fc_006': [
            ['x5', 'y4'],
            ['bp_006'],
            [0.944]
        ],
        'fc_007': [
            ['x5', 'y3'],
            ['bp_007'],
            [0.894]
        ],
        'fc_008': [
            ['x4', 'y3'],
            ['bp_008'],
            [0.941]
        ],
        'fc_009': [
            ['x3', 'y3'],
            ['bp_009'],
            [0.895]
        ],
        'pr_010': [
            ['x4', 'y1'],
            ['bp_021', 'bp_010', 'bp_011'],
            [0.862, 0.854, 0.945]
        ],
        'pr_011': [
            ['x6', 'y1'],
            ['bp_010', 'bp_011', 'bp_012'],
            [0.88, 0.864, 0.868]
        ],
        'pr_012': [
            ['x7', 'y2'],
            ['bp_011', 'bp_012', 'bp_013'],
            [0.898, 0.928, 0.858]
        ],
        'pr_013': [
            ['x7', 'y4'],
            ['bp_012', 'bp_013', 'bp_014'],
            [0.89, 0.894, 0.972]
        ],
        'pr_014': [
            ['x7', 'y6'],
            ['bp_013', 'bp_014', 'bp_015'],
            [0.987, 0.916, 0.857]
        ],
        'pr_015': [
            ['x6', 'y7'],
            ['bp_014', 'bp_015', 'bp_016'],
            [0.852, 0.908, 0.906]
        ],
        'pr_016': [
            ['x4', 'y7'],
            ['bp_015', 'bp_016', 'bp_017'],
            [0.967, 0.939, 0.982]
        ],
        'pr_017': [
            ['x2', 'y7'],
            ['bp_016', 'bp_017', 'bp_018'],
            [0.981, 0.915, 0.984]
        ],
        'pr_018': [
            ['x1', 'y6'],
            ['bp_017', 'bp_018', 'bp_019'],
            [0.934, 0.912, 0.903]
        ],
        'pr_019': [
            ['x1', 'y4'],
            ['bp_018', 'bp_019', 'bp_020'],
            [0.916, 0.873, 0.935]
        ],
        'pr_020': [
            ['x1', 'y2'],
            ['bp_019', 'bp_020', 'bp_021'],
            [0.943, 0.89, 0.94]
        ],
        'pr_021': [
            ['x2', 'y1'],
            ['bp_020', 'bp_021', 'bp_010'],
            [0.922, 0.934, 0.916]
        ],
        'bp_001': [
            ['gn_001'],
            [-0.672]
        ],
        'bp_002': [
            ['gn_002'],
            [0.686]
        ],
        'bp_003': [
            ['gn_003'],
            [0.711]
        ],
        'bp_004': [
            ['gn_004'],
            [-0.666]
        ],
        'bp_005': [
            ['gn_005'],
            [0.744]
        ],
        'bp_006': [
            ['gn_006'],
            [-0.967]
        ],
        'bp_007': [
            ['gn_007'],
            [0.883]
        ],
        'bp_008': [
            ['gn_008'],
            [0.684]
        ],
        'bp_009': [
            ['gn_009'],
            [-0.949]
        ],
        'bp_010': [
            ['gn_021', 'gn_010', 'gn_011'],
            [-0.873, -0.754, 0.866]
        ],
        'bp_011': [
            ['gn_010', 'gn_011', 'gn_012'],
            [0.957, 0.737, 0.987]
        ],
        'bp_012': [
            ['gn_011', 'gn_012', 'gn_013'],
            [-0.929, -0.979, -0.923]
        ],
        'bp_013': [
            ['gn_012', 'gn_013', 'gn_014'],
            [-0.861, -0.743, -0.887]
        ],
        'bp_014': [
            ['gn_013', 'gn_014', 'gn_015'],
            [0.94, 0.886, -0.671]
        ],
        'bp_015': [
            ['gn_014', 'gn_015', 'gn_016'],
            [0.842, 0.678, -0.729]
        ],
        'bp_016': [
            ['gn_015', 'gn_016', 'gn_017'],
            [0.806, 0.902, -0.795]
        ],
        'bp_017': [
            ['gn_016', 'gn_017', 'gn_018'],
            [-0.973, -0.978, 0.842]
        ],
        'bp_018': [
            ['gn_017', 'gn_018', 'gn_019'],
            [-0.819, 0.722, 0.959]
        ],
        'bp_019': [
            ['gn_018', 'gn_019', 'gn_020'],
            [0.81, -0.708, -0.951]
        ],
        'bp_020': [
            ['gn_019', 'gn_020', 'gn_021'],
            [-0.785, -0.807, 0.904]
        ],
        'bp_021': [
            ['gn_020', 'gn_021', 'gn_010'],
            [-0.876, 0.735, 0.765]
        ],
        'gn_001': [
            ['on_001'],
            [0.783]
        ],
        'gn_002': [
            ['on_002'],
            [0.841]
        ],
        'gn_003': [
            ['on_003'],
            [0.868]
        ],
        'gn_004': [
            ['on_004'],
            [0.906]
        ],
        'gn_005': [
            ['on_005'],
            [-0.872]
        ],
        'gn_006': [
            ['on_006'],
            [-0.677]
        ],
        'gn_007': [
            ['on_007'],
            [0.697]
        ],
        'gn_008': [
            ['on_008'],
            [-0.949]
        ],
        'gn_009': [
            ['on_009'],
            [0.796]
        ],
        'gn_010': [
            ['on_033', 'on_010', 'on_011', 'on_034'],
            [0.455, -0.758, 0.691]
        ],
        'gn_011': [
            ['on_010', 'on_011', 'on_012', 'on_035'],
            [0.402, 0.398, 0.453]
        ],
        'gn_012': [
            ['on_011', 'on_012', 'on_013', 'on_035'],
            [0.5, 0.634, 0.631]
        ],
        'gn_013': [
            ['on_012', 'on_013', 'on_014', 'on_036'],
            [-0.312, -0.378, -0.506]
        ],
        'gn_014': [
            ['on_013', 'on_014', 'on_015', 'on_037'],
            [-0.286, 0.273, -0.517]
        ],
        'gn_015': [
            ['on_014', 'on_015', 'on_016', 'on_037'],
            [-0.213, 0.632, 0.401]
        ],
        'gn_016': [
            ['on_015', 'on_016', 'on_017', 'on_038'],
            [0.69, 0.296, 0.275]
        ],
        'gn_017': [
            ['on_016', 'on_017', 'on_018', 'on_039'],
            [0.552, 0.597, -0.428]
        ],
        'gn_018': [
            ['on_017', 'on_018', 'on_019', 'on_039'],
            [-0.417, -0.632, 0.677]
        ],
        'gn_019': [
            ['on_018', 'on_019', 'on_020', 'on_040'],
            [-0.631, -0.52, 0.748]
        ],
        'gn_020': [
            ['on_019', 'on_020', 'on_021', 'on_041'],
            [0.218, -0.46, -0.213]
        ],
        'gn_021': [
            ['on_020', 'on_021', 'on_022', 'on_041'],
            [0.606, 0.513, -0.566]
        ],
        'on_001': [
            ['dd_001'],
            [0.379]
        ],
        'on_002': [
            ['dd_001', 'dd_002'],
            [-0.778, -0.21]
        ],
        'on_003': [
            ['dd_001', 'dd_002', 'dd_003', 'dd_004'],
            [-0.604, 0.485, -0.683, 0.773]
        ],
        'on_004': [
            ['dd_001', 'dd_004'],
            [-0.366, 0.313]
        ],
        'on_005': [
            ['dd_001', 'dd_004', 'dd_005', 'dd_006'],
            [-0.384, -0.313, 0.497, -0.544]
        ],
        'on_006': [
            ['dd_001', 'dd_006'],
            [-0.291, 0.649]
        ],
        'on_007': [
            ['dd_001', 'dd_006', 'dd_007', 'dd_008'],
            [-0.531, 0.301, 0.795, 0.201]
        ],
        'on_008': [
            ['dd_001', 'dd_008'],
            [-0.615, -0.602]
        ],
        'on_009': [
            ['dd_001', 'dd_002', 'dd_008', 'dd_009'],
            [0.306, 0.393, 0.422, -0.64]
        ],
        'on_010': [
            ['dd_002'],
            [-0.547]
        ],
        'on_011': [
            ['dd_002', 'dd_003'],
            [0.481, -0.561]
        ],
        'on_012': [
            ['dd_003'],
            [0.381]
        ],
        'on_013': [
            ['dd_003'],
            [-0.464]
        ],
        'on_014': [
            ['dd_003'],
            [0.6]
        ],
        'on_015': [
            ['dd_003', 'dd_004'],
            [0.547, -0.31]
        ],
        'on_016': [
            ['dd_004'],
            [-0.662]
        ],
        'on_017': [
            ['dd_004', 'dd_005'],
            [-0.621, -0.358]
        ],
        'on_018': [
            ['dd_005'],
            [0.8]
        ],
        'on_019': [
            ['dd_005'],
            [-0.535]
        ],
        'on_020': [
            ['dd_005'],
            [0.616]
        ],
        'on_021': [
            ['dd_005', 'dd_006'],
            [0.228, -0.413]
        ],
        'on_022': [
            ['dd_006'],
            [0.416]
        ],
        'on_023': [
            ['dd_006', 'dd_007'],
            [-0.655, -0.695]
        ],
        'on_024': [
            ['dd_007'],
            [0.308]
        ],
        'on_025': [
            ['dd_007'],
            [-0.34]
        ],
        'on_026': [
            ['dd_007'],
            [0.698]
        ],
        'on_027': [
            ['dd_007', 'dd_008'],
            [0.416, -0.639]
        ],
        'on_028': [
            ['dd_008'],
            [0.362]
        ],
        'on_029': [
            ['dd_008', 'dd_009'],
            [-0.234, 0.48]
        ],
        'on_030': [
            ['dd_009'],
            [0.299]
        ],
        'on_031': [
            ['dd_009'],
            [0.739]
        ],
        'on_032': [
            ['dd_009'],
            [-0.555]
        ],
        'on_033': [
            ['dd_002', 'dd_009'],
            [0.57, -0.222]
        ],
        'on_034': [
            ['dd_002'],
            [-0.724]
        ],
        'on_035': [
            ['dd_003'],
            [-0.278]
        ],
        'on_036': [
            ['dd_004'],
            [0.377]
        ],
        'on_037': [
            ['dd_005'],
            [0.477]
        ],
        'on_038': [
            ['dd_006'],
            [0.491]
        ],
        'on_039': [
            ['dd_007'],
            [0.205]
        ],
        'on_040': [
            ['dd_008'],
            [-0.464]
        ],
        'on_041': [
            ['dd_009'],
            [-0.651]
        ]
    }

    # ONN Brain Map
    # brain_file = open('success_post_synaptic_neighbors_dictionary.json', 'r')
    # contents = brain_file.read()
    # starting_post_synaptic_neighbors_dictionary = ast.literal_eval(contents)
    # brain_file.close()

    main()

