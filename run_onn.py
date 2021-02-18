

import time
from PIL import Image
import concurrent.futures


X_COORD = 7 -1         # '-1' for 0-th Index
Y_COORD = 7 -1         # '-1' for 0-th Index
EPOCHS = 6             # Number of times to run training

''' The Neural Network '''
# Standard Use Dictionary for Testing
starting_post_synaptic_neighbors_dictionary = {   # neuron_to_post-synaptic

    'fc_001': [
        ['bp_001'],
        [0.876], 
        ['x4', 'y4'],
    ],
    'fc_002': [
        ['bp_002'],
        [0.853],
        ['x4', 'y3'],
    ],
    'fc_003': [
        ['bp_003'],
        [0.952],
        ['x5', 'y3'],
    ],
    'fc_004': [
        ['bp_004'],
        [0.869],
        ['x5', 'y4'],
    ],
    'fc_005': [
        ['bp_005'],
        [0.977],
        ['x5', 'y5'],
    ],
    'fc_006': [
        ['bp_006'],
        [0.944],
        ['x5', 'y4'],
    ],
    'fc_007': [
        ['bp_007'],
        [0.894],
        ['x5', 'y3'],
    ],
    'fc_008': [
        ['bp_008'],
        [0.941],
        ['x4', 'y3'],
    ],
    'fc_009': [
        ['bp_009'],
        [0.895],
        ['x3', 'y3'],
    ],
    'pr_010': [
        ['bp_021', 'bp_010', 'bp_011'],
        [0.862, 0.854, 0.945],
        ['x4', 'y1'],
    ],
    'pr_011': [
        ['bp_010', 'bp_011', 'bp_012'],
        [0.88, 0.864, 0.868],
        ['x6', 'y1'],
    ],
    'pr_012': [
        ['bp_011', 'bp_012', 'bp_013'],
        [0.898, 0.928, 0.858],
        ['x7', 'y2'],
    ],
    'pr_013': [
        ['bp_012', 'bp_013', 'bp_014'],
        [0.89, 0.894, 0.972],
        ['x7', 'y4'],
    ],
    'pr_014': [
        ['bp_013', 'bp_014', 'bp_015'],
        [0.987, 0.916, 0.857],
        ['x7', 'y6'],
    ],
    'pr_015': [
        ['bp_014', 'bp_015', 'bp_016'],
        [0.852, 0.908, 0.906],
        ['x6', 'y7'],
    ],
    'pr_016': [
        ['bp_015', 'bp_016', 'bp_017'],
        [0.967, 0.939, 0.982],
        ['x4', 'y7'],
    ],
    'pr_017': [
        ['bp_016', 'bp_017', 'bp_018'],
        [0.981, 0.915, 0.984],
        ['x2', 'y7'],
    ],
    'pr_018': [
        ['bp_017', 'bp_018', 'bp_019'],
        [0.934, 0.912, 0.903],
        ['x1', 'y6'],
    ],
    'pr_019': [
        ['bp_018', 'bp_019', 'bp_020'],
        [0.916, 0.873, 0.935],
        ['x1', 'y4'],
    ],
    'pr_020': [
        ['bp_019', 'bp_020', 'bp_021'],
        [0.943, 0.89, 0.94],
        ['x1', 'y2'],
    ],
    'pr_021': [
        ['bp_020', 'bp_021', 'bp_010'],
        [0.922, 0.934, 0.916],
        ['x2', 'y1'],
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
    ],
    'dd_001': [
        [0,0],
    ],
    'dd_002': [
        [0,1],
    ],
    'dd_003': [
        [1,1],
    ],
    'dd_004': [
        [1,0],
    ],
    'dd_005': [
        [1,-1],
    ],
    'dd_006': [
        [0,-1],
    ],
    'dd_007': [
        [-1,-1],
    ],
    'dd_008': [
        [-1,0],
    ],
    'dd_009': [
        [-1,1],
    ],
}


''' Read the Image '''
img = Image.open("img_triangle_169.png").convert('LA')
height, width = img.size
total_rows = height
total_colums = width
sequence_of_pixels = img.getdata()
list_of_pixel_tuples = list(sequence_of_pixels)

print("\n\n\n_____ START _____\n")
print(f"Height x Width: {height}px x {width}px\nTotal # of Pixels: {len(list_of_pixel_tuples)}\n")

pixel_list = []
for pixel in list_of_pixel_tuples:
    pixel_row = []   # temporary list of pixels in row
    pixels = list(pixel)   # convert tuple to list
    p_normalized = round(pixels.pop(0)/255, 6)   # convert 0-255 to 0-1
    p = round(1 - p_normalized, 3)   # convert white to 0 and black to 1
    pixel_list.append(p)

pixel_matrix = []
row=1
column_count=1

print("pixel_matrix = [")
for pixel in pixel_list:
    if column_count == 1:
        temp_row = []
        temp_row.append(pixel)
        column_count+=1
    elif column_count == width:
        temp_row.append(pixel)
        print(" ", temp_row)
        pixel_matrix.append(temp_row)
        row+=1
        column_count=1
    else:
        temp_row.append(pixel)
        column_count+=1
print("]")


''' All the Functions '''
# Generate Signal through 1 Neuron
def generate_signal(neuron_from_signal_dictionary):   # nucleus

    neuron = neuron_from_signal_dictionary   # neuron id
    neuron_id = neuron
    # print("signal_dictionary - gen func:", signal_dictionary)
    signal_input = signal_dictionary[neuron]   # neuron's signal

    signal = sum(signal_input)

    post_synaptic_neighbors = starting_post_synaptic_neighbors_dictionary[neuron][0]
    synapse_values = starting_post_synaptic_neighbors_dictionary[neuron][1]

    signal_outputs = [synapse_value * signal for synapse_value in synapse_values]   # multiply signal by array of synapse_values

    return signal_outputs, post_synaptic_neighbors, neuron_id, signal

# Use the Signal to determine which direction to move
def chose_direction(neurons_firing):

    direction_neuron = None
    highest_signal = 0

    for neuron in neurons_firing:
        
        signal_sum = sum(signal_dictionary[neuron])
        # direction_dictionary[neuron] = [[len(signal_dictionary[neuron])], [sum(signal_dictionary[neuron])]]

        if signal_sum > highest_signal:
            direction_neuron = neuron
            highest_signal = signal_sum
        else:
            pass

    print("\nDirection & Signal:", direction_neuron, highest_signal)

    new_x_move = direction_lookup[direction_neuron][0]
    new_y_move = direction_lookup[direction_neuron][1]

    return new_x_move, new_y_move


def main():

    t = 1
    x_coord = X_COORD
    y_coord = Y_COORD

    for _ in range(EPOCHS):

        ''' Signal Loop '''
        while signal_dictionary:

            if 'dd_001' in signal_dictionary:
                new_x_move, new_y_move = chose_direction(list(signal_dictionary.keys()))
                print(f"New_x_move, New_y_move: {new_x_move}x, {new_y_move}y")
                signal_dictionary.clear()               

            else:
                # Run concurrent neuron firings
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = {executor.submit(generate_signal, neuron) for neuron in signal_dictionary}

                    new_signal_dict = {}
                    update_neuron_dict = {}
                    for neuron in concurrent.futures.as_completed(futures):
                        signal_outputs, post_synaptic_neighbors, neuron_id, signal = neuron.result()

                        for neighbor, signal_output in zip(post_synaptic_neighbors, signal_outputs):
                            new_signal_dict.setdefault(neighbor,[]).append(signal_output)
                        update_neuron_dict[neuron_id] = signal

                # Update signasignal_dictionaryl_dictionary with which neurons to fire next
                signal_dictionary.clear()
                signal_dictionary.update(new_signal_dict)
                print("\nUPDATED SIGNAL PATH:", signal_dictionary)
                time.sleep(1)

        ''' UPDATE MOVEMENT DIRECTION '''
        x_move=new_x_move
        y_move=new_y_move
        x_coord+=x_move
        y_coord+=y_move
        print(f"\n\nMOVING FOCUS... NEW TARGET: {x_coord}x, {y_coord}y")

        # Get New Field of Vision Inputs
        fc_001 = pixel_matrix[y_coord][x_coord]
        fc_002 = pixel_matrix[y_coord-1][x_coord]
        fc_003 = pixel_matrix[y_coord-1][x_coord+1]
        fc_004 = pixel_matrix[y_coord][x_coord+1]
        fc_005 = pixel_matrix[y_coord+1][x_coord+1]
        fc_006 = pixel_matrix[y_coord+1][x_coord]
        fc_007 = pixel_matrix[y_coord+1][x_coord-1]
        fc_008 = pixel_matrix[y_coord][x_coord-1]
        fc_009 = pixel_matrix[y_coord-1][x_coord-1]
        try:        
            pr_010 = pixel_matrix[y_coord-3][x_coord]
        except IndexError:
            pr_010 = 0.000
        try:
            pr_011 = pixel_matrix[y_coord-3][x_coord+2]
        except IndexError:
            pr_011 = 0.000
        try:
            pr_012 = pixel_matrix[y_coord-2][x_coord+3]
        except IndexError:
            pr_012 = 0.000
        try:
            pr_013 = pixel_matrix[y_coord][x_coord+3]
        except IndexError:
            pr_013 = 0.000
        try:
            pr_014 = pixel_matrix[y_coord+2][x_coord+3]
        except IndexError:
            pr_014 = 0.000
        try:
            pr_015 = pixel_matrix[y_coord+3][x_coord+2]
        except IndexError:
            pr_015 = 0.000
        try:
            pr_016 = pixel_matrix[y_coord+3][x_coord]
        except IndexError:
            pr_016 = 0.000
        try:
            pr_017 = pixel_matrix[y_coord+3][x_coord-2]
        except IndexError:
            pr_017 = 0.000
        try:
            pr_018 = pixel_matrix[y_coord+2][x_coord-3]
        except IndexError:
            pr_018 = 0.000
        try:
            pr_019 = pixel_matrix[y_coord][x_coord-3]
        except IndexError:
            pr_019 = 0.000
        try:
            pr_020 = pixel_matrix[y_coord-2][x_coord-3]
        except IndexError:
            pr_020 = 0.000
        try:
            pr_021 = pixel_matrix[y_coord-3][x_coord-2]
        except IndexError:
            pr_021 = 0.000

        # Print what the AI is seeing
        try:
            print(f"\n\ntimestep: t{t}")
            print(f"center_of_fv: fv_x{x_coord}_y{y_coord}")
            print("cofv_signal:", pixel_matrix[y_coord][x_coord])
            print("pv_nrth:", "[-------]", [format(pr_021, '.3f')], "[-------]", [format(pr_010, '.3f')], "[-------]", [format(pr_011, '.3f')], "[-------]")
            print("pv_nrth:", [format(pr_020, '.3f')], "[-------]", "[-------]", "[-------]", "[-------]", "[-------]", [format(pr_012, '.3f')])
            print("fv_view:", "[-------]", "[-------]", [format(fc_009, '.3f')], [format(fc_002, '.3f')], [format(fc_003, '.3f')], "[-------]", "[-------]")
            print("        ", [format(pr_019, '.3f')], "[-------]", [format(fc_008, '.3f')], [format(fc_001, '.3f')], [format(fc_004, '.3f')], "[-------]", [format(pr_013, '.3f')])
            print("        ", "[-------]", "[-------]", [format(fc_007, '.3f')], [format(fc_006, '.3f')], [format(fc_005, '.3f')], "[-------]", "[-------]") 
            print("pv_sth: ", [format(pr_018, '.3f')], "[-------]", "[-------]", "[-------]", "[-------]", "[-------]", [format(pr_014, '.3f')])
            print("pv_sth: ", "[-------]", [format(pr_017, '.3f')], "[-------]", [format(pr_016, '.3f')], "[-------]", [format(pr_015, '.3f')], "[-------]")
        except IndexError:
            print("pv_out: ", "[xxxxxxx]", "[xxxxxxx]", "[xxxxxxx]", "[xxxxxxx]", "[xxxxxxx]", "[xxxxxxx]", "[xxxxxxx]")

        new_signal_dictionary = {                       # Put in NEW Signal Dictionary
            'fc_002': [fc_002],
            'fc_001': [fc_001],
            'fc_003': [fc_003],
            'fc_004': [fc_004],
            'fc_005': [fc_005],
            'fc_006': [fc_006],
            'fc_007': [fc_007],
            'fc_008': [fc_008],
            'fc_009': [fc_009],
            'pr_010': [pr_010],
            'pr_011': [pr_011],
            'pr_012': [pr_012],
            'pr_013': [pr_013],
            'pr_014': [pr_014],
            'pr_015': [pr_015],
            'pr_016': [pr_016],
            'pr_017': [pr_017],
            'pr_018': [pr_018],
            'pr_019': [pr_019],
            'pr_020': [pr_020],
            'pr_021': [pr_021],
        }
        signal_dictionary.update(new_signal_dictionary)         # Update 'signal_dictionary' w/ NEW Signal Dictionary
        
        print(f"\n\nNEW SIGNAL DICTIONARY: {signal_dictionary}")

        t+=1





""" RUN """
if __name__ == "__main__":

    t = 0
    x_coord = X_COORD
    y_coord = Y_COORD


    # Get New Field of Vision Inputs
    fc_001 = pixel_matrix[y_coord][x_coord]
    fc_002 = pixel_matrix[y_coord-1][x_coord]
    fc_003 = pixel_matrix[y_coord-1][x_coord+1]
    fc_004 = pixel_matrix[y_coord][x_coord+1]
    fc_005 = pixel_matrix[y_coord+1][x_coord+1]
    fc_006 = pixel_matrix[y_coord+1][x_coord]
    fc_007 = pixel_matrix[y_coord+1][x_coord-1]
    fc_008 = pixel_matrix[y_coord][x_coord-1]
    fc_009 = pixel_matrix[y_coord-1][x_coord-1]
    pr_010 = pixel_matrix[y_coord-3][x_coord]
    pr_011 = pixel_matrix[y_coord-3][x_coord+2]
    pr_012 = pixel_matrix[y_coord-2][x_coord+3]
    pr_013 = pixel_matrix[y_coord][x_coord+3]
    pr_014 = pixel_matrix[y_coord+2][x_coord+3]
    pr_015 = pixel_matrix[y_coord+3][x_coord+2]
    pr_016 = pixel_matrix[y_coord+3][x_coord]
    pr_017 = pixel_matrix[y_coord+3][x_coord-2]
    pr_018 = pixel_matrix[y_coord+2][x_coord-3]
    pr_019 = pixel_matrix[y_coord][x_coord-3]
    pr_020 = pixel_matrix[y_coord-2][x_coord-3]
    pr_021 = pixel_matrix[y_coord-3][x_coord-2]

    print(pixel_matrix[y_coord])

    # Print what the AI is seeing
    try:
        print(f"\n\ntimestep: t{t}")
        print(f"center_of_fv: fv_x{x_coord}_y{y_coord}")
        print("cofv_signal:", pixel_matrix[y_coord][x_coord])
        print("pv_nrth:", "[-------]", [format(pr_021, '.3f')], "[-------]", [format(pr_010, '.3f')], "[-------]", [format(pr_011, '.3f')], "[-------]")
        print("pv_nrth:", [format(pr_020, '.3f')], "[-------]", "[-------]", "[-------]", "[-------]", "[-------]", [format(pr_012, '.3f')])
        print("fv_view:", "[-------]", "[-------]", [format(fc_009, '.3f')], [format(fc_002, '.3f')], [format(fc_003, '.3f')], "[-------]", "[-------]")
        print("        ", [format(pr_019, '.3f')], "[-------]", [format(fc_008, '.3f')], [format(fc_001, '.3f')], [format(fc_004, '.3f')], "[-------]", [format(pr_013, '.3f')])
        print("        ", "[-------]", "[-------]", [format(fc_007, '.3f')], [format(fc_006, '.3f')], [format(fc_005, '.3f')], "[-------]", "[-------]") 
        print("pv_sth: ", [format(pr_018, '.3f')], "[-------]", "[-------]", "[-------]", "[-------]", "[-------]", [format(pr_014, '.3f')])
        print("pv_sth: ", "[-------]", [format(pr_017, '.3f')], "[-------]", [format(pr_016, '.3f')], "[-------]", [format(pr_015, '.3f')], "[-------]")
    except IndexError:
        print("pv_out: ", "[xxxxxxx]", "[xxxxxxx]", "[xxxxxxx]", "[xxxxxxx]", "[xxxxxxx]", "[xxxxxxx]", "[xxxxxxx]")

    signal_dictionary = {   # set the signal
        'fc_001': [pixel_matrix[y_coord][x_coord]],
        'fc_002': [pixel_matrix[y_coord-1][x_coord]],
        'fc_003': [pixel_matrix[y_coord-1][x_coord+1]],
        'fc_004': [pixel_matrix[y_coord][x_coord+1]],
        'fc_005': [pixel_matrix[y_coord+1][x_coord+1]],
        'fc_006': [pixel_matrix[y_coord+1][x_coord]],
        'fc_007': [pixel_matrix[y_coord+1][x_coord-1]],
        'fc_008': [pixel_matrix[y_coord][x_coord-1]],
        'fc_009': [pixel_matrix[y_coord-1][x_coord-1]],
        'pr_010': [pixel_matrix[y_coord-3][x_coord]],
        'pr_011': [pixel_matrix[y_coord-3][x_coord+2]],
        'pr_012': [pixel_matrix[y_coord-2][x_coord+3]],
        'pr_013': [pixel_matrix[y_coord][x_coord+3]],
        'pr_014': [pixel_matrix[y_coord+2][x_coord+3]],
        'pr_015': [pixel_matrix[y_coord+3][x_coord+2]],
        'pr_016': [pixel_matrix[y_coord+3][x_coord]],
        'pr_017': [pixel_matrix[y_coord+3][x_coord-2]],
        'pr_018': [pixel_matrix[y_coord+2][x_coord-3]],
        'pr_019': [pixel_matrix[y_coord][x_coord-3]],
        'pr_020': [pixel_matrix[y_coord-2][x_coord-3]],
        'pr_021': [pixel_matrix[y_coord-3][x_coord-2]],
    }

    direction_lookup = {    
        'dd_001': 
            [0,0],
        'dd_002': 
            [0,1],
        'dd_003': 
            [1,1],
        'dd_004': 
            [1,0],
        'dd_005': 
            [1,-1],
        'dd_006': 
            [0,-1],
        'dd_007': 
            [-1,-1],
        'dd_008': 
            [-1,0],
        'dd_009': 
            [-1,1],
    }

    main()
    