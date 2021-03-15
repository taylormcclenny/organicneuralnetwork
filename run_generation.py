

import time
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from PIL import Image
import concurrent.futures


SHOW_VF_IMAGE = 'OFF'
PAUSE_TIME = 0.25

ACTIVATION_THRESHOLD = 0      # For now, all neurons share this AT. Ultimately it should be neuron-specific & self-tuned
SLEEP_TIME = 0.1


''' Helper Dictionaries & Lists '''
mutation_list = ['gen_1_mutation_5', 'gen_1_mutation_6', 'gen_1_mutation_7']
shape_list = ['triangle', 'circle', 'square']
starting_positions = [(5,5), (14,5), (5,14), (14,14)]

# Shape Specific Neuron list for calculating classification
shape_specific_neurons = ['ss_001', 'ss_002', 'ss_003', 'ss_004', 'ss_005', 'ss_006', 'ss_007', 'ss_008', 'ss_009', 'ss_010', 'ss_011', 'ss_012']

ganglion_neighbors_dictionary = {           # Ganglion's Spacial-Neighboring Bipolar Cells
    'gn_001' : ['bp_003', 'bp_009', 'bp_013', 'bp_014', 'bp_015', 'bp_016', 'bp_017', 'bp_018', 'bp_019', 'bp_020', 'bp_021'],
    'gn_002' : ['bp_001', 'bp_009', 'bp_007', 'bp_022', 'bp_013', 'bp_014', 'bp_015', 'bp_016', 'bp_017', 'bp_018', 'bp_019', 'bp_020', 'bp_021'],
    'gn_003' : ['bp_003', 'bp_022', 'bp_013', 'bp_014', 'bp_015', 'bp_016', 'bp_017', 'bp_018', 'bp_019', 'bp_020', 'bp_021'],
    'gn_004' : ['bp_001', 'bp_003', 'bp_026', 'bp_028', 'bp_013', 'bp_014', 'bp_015', 'bp_016', 'bp_017', 'bp_018', 'bp_019', 'bp_020', 'bp_021'],
    'gn_005' : ['bp_002', 'bp_004', 'bp_005', 'bp_010', 'bp_011', 'bp_014', 'bp_016', 'bp_017'],
    'gn_006' : ['bp_004', 'bp_005', 'bp_006', 'bp_013', 'bp_015', 'bp_016', 'bp_017', 'bp_018'],
    'gn_007' : ['bp_005', 'bp_006', 'bp_008', 'bp_014', 'bp_017', 'bp_018', 'bp_023', 'bp_024'],
    'gn_008' : ['bp_010', 'bp_011', 'bp_012', 'bp_013', 'bp_014', 'bp_017', 'bp_019', 'bp_020'],
    'gn_009' : ['bp_013', 'bp_014', 'bp_015', 'bp_016', 'bp_018', 'bp_019', 'bp_020', 'bp_021'],
    'gn_010' : ['bp_014', 'bp_015', 'bp_017', 'bp_020', 'bp_021', 'bp_023', 'bp_024', 'bp_025'],
    'gn_011' : ['bp_011', 'bp_012', 'bp_016', 'bp_017', 'bp_020', 'bp_027', 'bp_029', 'bp_030'],
    'gn_012' : ['bp_016', 'bp_017', 'bp_018', 'bp_019', 'bp_021', 'bp_029', 'bp_030', 'bp_031'],
    'gn_013' : ['bp_017', 'bp_018', 'bp_020', 'bp_024', 'bp_025', 'bp_030', 'bp_031', 'bp_033'],
    'gn_014' : ['bp_003', 'bp_007', 'bp_028', 'bp_032', 'bp_013', 'bp_014', 'bp_015', 'bp_016', 'bp_017', 'bp_018', 'bp_019', 'bp_020', 'bp_021'],
    'gn_015' : ['bp_009', 'bp_028', 'bp_013', 'bp_014', 'bp_015', 'bp_016', 'bp_017', 'bp_018', 'bp_019', 'bp_020', 'bp_021'],
    'gn_016' : ['bp_009', 'bp_022', 'bp_026', 'bp_032', 'bp_013', 'bp_014', 'bp_015', 'bp_016', 'bp_017', 'bp_018', 'bp_019', 'bp_020', 'bp_021'],
    'gn_017' : ['bp_028', 'bp_022', 'bp_013', 'bp_014', 'bp_015', 'bp_016', 'bp_017', 'bp_018', 'bp_019', 'bp_020', 'bp_021'],
}
bp_blacklist = ['bp_002', 'bp_004', 'bp_005', 'bp_006', 'bp_008', 'bp_010', 'bp_011', 'bp_012', 'bp_023', 'bp_024', 'bp_025', 'bp_027', 'bp_029', 'bp_030', 'bp_031', 'bp_033']
dd_ganglion_dictionary = {           # dd
    'dd_001' : ['gn_001'],
    'dd_002' : ['gn_002'],
    'dd_003' : ['gn_003'],
    'dd_004' : ['gn_004'],
    'dd_005' : ['gn_005', 'gn_006', 'gn_007', 'gn_008', 'gn_009', 'gn_010', 'gn_011', 'gn_012', 'gn_013'],
    'dd_006' : ['gn_014'],
    'dd_007' : ['gn_015'],
    'dd_008' : ['gn_016'],
    'dd_009' : ['gn_017'],
}
direction_lookup = {    
    'dd_001': 
        [-1,-1],
    'dd_002': 
        [0,-1],
    'dd_003': 
        [1,-1],
    'dd_004': 
        [-1,0],
    'dd_005': 
        [0,0],
    'dd_006': 
        [1,0],
    'dd_007': 
        [-1,1],
    'dd_008': 
        [0,1],
    'dd_009': 
        [1,1],
}
bipolar_value_dictionary = {}
ganglion_n_bpv_dictionary = {}
gn_direction_deciding_inputs = {}
dd_signal_dictionary = {}


def main():

    mutation_performance = {}
    for mutation in mutation_list:
        
        global starting_post_synaptic_neighbors_dictionary

        ''' Load the Neural Network '''
        file_name = f'onn_map_{mutation}'
        with open(f'mutations/{file_name}.json', 'r') as file:
            starting_post_synaptic_neighbors_dictionary = file.read()
        # Convert JSON string to Python Dictionary
        starting_post_synaptic_neighbors_dictionary = ast.literal_eval(starting_post_synaptic_neighbors_dictionary)

        shape_prediction_performance_list = []
        for shape in shape_list:
            
            ''' Read the Image '''
            img = Image.open(f"img_{shape}_324.png").convert('LA')
            height, width = img.size
            # total_rows = height
            # total_colums = width
            sequence_of_pixels = img.getdata()
            list_of_pixel_tuples = list(sequence_of_pixels)

            print("\n\n\n_____ START _____\n")
            print(f"Height x Width: {height}px x {width}px\nTotal # of Pixels: {len(list_of_pixel_tuples)}\n")

            pixel_list = []
            for pixel in list_of_pixel_tuples:
                # pixel_row = []   # temporary list of pixels in row
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


            for starting_position in starting_positions:

                global signal_dictionary

                t = 0
                x_coord = starting_position[0] - 1      # For reading index of Image Pixel Matrix (List of lists)
                y_coord = starting_position[1] - 1      # (!) At sampling COORD is subtracted by 1 ('-1') due to referencing 0-th Indexed List

                path_dictionary = {}
                path_dictionary[t] = [x_coord+1, y_coord+1]

                px_001 = pixel_matrix[y_coord-4][x_coord-4]         # Section 1 - Top Left
                px_002 = pixel_matrix[y_coord-4][x_coord-3]
                px_003 = pixel_matrix[y_coord-4][x_coord-2]
                px_004 = pixel_matrix[y_coord-3][x_coord-4]
                px_005 = pixel_matrix[y_coord-3][x_coord-3]
                px_006 = pixel_matrix[y_coord-3][x_coord-2]
                px_007 = pixel_matrix[y_coord-2][x_coord-4]
                px_008 = pixel_matrix[y_coord-2][x_coord-3]
                px_009 = pixel_matrix[y_coord-2][x_coord-2]

                px_010 = pixel_matrix[y_coord-4][x_coord-1]         # Section 2 - Top Middle
                px_011 = pixel_matrix[y_coord-4][x_coord+0]         
                px_012 = pixel_matrix[y_coord-4][x_coord+1]
                px_013 = pixel_matrix[y_coord-3][x_coord-1] 
                px_014 = pixel_matrix[y_coord-3][x_coord+0] 
                px_015 = pixel_matrix[y_coord-3][x_coord+1]
                px_016 = pixel_matrix[y_coord-2][x_coord-1] 
                px_017 = pixel_matrix[y_coord-2][x_coord+0] 
                px_018 = pixel_matrix[y_coord-2][x_coord+1]

                px_019 = pixel_matrix[y_coord-4][x_coord+2]         # Section 3 - Top Right
                px_020 = pixel_matrix[y_coord-4][x_coord+3]
                px_021 = pixel_matrix[y_coord-4][x_coord+4]
                px_022 = pixel_matrix[y_coord-3][x_coord+2]
                px_023 = pixel_matrix[y_coord-3][x_coord+3]
                px_024 = pixel_matrix[y_coord-3][x_coord+4]
                px_025 = pixel_matrix[y_coord-2][x_coord+2]         
                px_026 = pixel_matrix[y_coord-2][x_coord+3]             
                px_027 = pixel_matrix[y_coord-2][x_coord+4]

                px_028 = pixel_matrix[y_coord-1][x_coord-4]         # Section 4 - Middle Left
                px_029 = pixel_matrix[y_coord-1][x_coord-3] 
                px_030 = pixel_matrix[y_coord-1][x_coord-2] 
                px_031 = pixel_matrix[y_coord+0][x_coord-4] 
                px_032 = pixel_matrix[y_coord+0][x_coord-3] 
                px_033 = pixel_matrix[y_coord+0][x_coord-2] 
                px_034 = pixel_matrix[y_coord+1][x_coord-4] 
                px_035 = pixel_matrix[y_coord+1][x_coord-3] 
                px_036 = pixel_matrix[y_coord+1][x_coord-2]

                px_037 = pixel_matrix[y_coord-1][x_coord-1]         # Section 5 - Center
                px_038 = pixel_matrix[y_coord-1][x_coord+0]
                px_039 = pixel_matrix[y_coord-1][x_coord+1]
                px_040 = pixel_matrix[y_coord+0][x_coord-1] 
                px_041 = pixel_matrix[y_coord+0][x_coord+0]         ### Middle of Focal Vision
                px_042 = pixel_matrix[y_coord+0][x_coord+1]
                px_043 = pixel_matrix[y_coord+1][x_coord-1] 
                px_044 = pixel_matrix[y_coord+1][x_coord+0]
                px_045 = pixel_matrix[y_coord+1][x_coord+1]
                
                px_046 = pixel_matrix[y_coord-1][x_coord+2]         # Section 6 - Middle Right
                px_047 = pixel_matrix[y_coord-1][x_coord+3]
                px_048 = pixel_matrix[y_coord-1][x_coord+4]
                px_049 = pixel_matrix[y_coord+0][x_coord+2]
                px_050 = pixel_matrix[y_coord+0][x_coord+3]
                px_051 = pixel_matrix[y_coord+0][x_coord+4]
                px_052 = pixel_matrix[y_coord+1][x_coord+2]         
                px_053 = pixel_matrix[y_coord+1][x_coord+3]             
                px_054 = pixel_matrix[y_coord+1][x_coord+4] 

                px_055 = pixel_matrix[y_coord+2][x_coord-4]         # Section 7 - Bottom Left
                px_056 = pixel_matrix[y_coord+2][x_coord-3]
                px_057 = pixel_matrix[y_coord+2][x_coord-2]
                px_058 = pixel_matrix[y_coord+3][x_coord-4]
                px_059 = pixel_matrix[y_coord+3][x_coord-3]
                px_060 = pixel_matrix[y_coord+3][x_coord-2]
                px_061 = pixel_matrix[y_coord+4][x_coord-4]
                px_062 = pixel_matrix[y_coord+4][x_coord-3]
                px_063 = pixel_matrix[y_coord+4][x_coord-2]

                px_064 = pixel_matrix[y_coord+2][x_coord-1]         # Section 8 - Bottom Middle
                px_065 = pixel_matrix[y_coord+2][x_coord+0]         
                px_066 = pixel_matrix[y_coord+2][x_coord+1]
                px_067 = pixel_matrix[y_coord+3][x_coord-1] 
                px_068 = pixel_matrix[y_coord+3][x_coord+0] 
                px_069 = pixel_matrix[y_coord+3][x_coord+1]
                px_070 = pixel_matrix[y_coord+4][x_coord-1] 
                px_071 = pixel_matrix[y_coord+4][x_coord+0] 
                px_072 = pixel_matrix[y_coord+4][x_coord+1]

                px_073 = pixel_matrix[y_coord+2][x_coord+2]         # Section 9 - Bottom Right
                px_074 = pixel_matrix[y_coord+2][x_coord+3]
                px_075 = pixel_matrix[y_coord+2][x_coord+4]
                px_076 = pixel_matrix[y_coord+3][x_coord+2]
                px_077 = pixel_matrix[y_coord+3][x_coord+3]
                px_078 = pixel_matrix[y_coord+3][x_coord+4]
                px_079 = pixel_matrix[y_coord+4][x_coord+2]         
                px_080 = pixel_matrix[y_coord+4][x_coord+3]             
                px_081 = pixel_matrix[y_coord+4][x_coord+4]

                # Print what the AI is seeing
                try:
                    print(f"\n\ntimestep: t{t}")
                    print(f"center_of_fv: fv_x{x_coord+1}_y{y_coord+1}")
                    print("cofv_signal:", pixel_matrix[y_coord][x_coord])
                    print("periph :", [format(px_001, '.3f')], [format(px_002, '.3f')], [format(px_003, '.3f')], [format(px_010, '.3f')], [format(px_011, '.3f')], [format(px_012, '.3f')], [format(px_019, '.3f')], [format(px_020, '.3f')], [format(px_021, '.3f')])
                    print("periph :", [format(px_004, '.3f')], [format(px_005, '.3f')], [format(px_006, '.3f')], [format(px_013, '.3f')], [format(px_014, '.3f')], [format(px_015, '.3f')], [format(px_022, '.3f')], [format(px_023, '.3f')], [format(px_024, '.3f')])
                    print("periph :", [format(px_007, '.3f')], [format(px_008, '.3f')], [format(px_009, '.3f')], [format(px_016, '.3f')], [format(px_017, '.3f')], [format(px_018, '.3f')], [format(px_025, '.3f')], [format(px_026, '.3f')], [format(px_027, '.3f')])
                    print("focal  :", [format(px_028, '.3f')], [format(px_029, '.3f')], [format(px_030, '.3f')], [format(px_037, '.3f')], [format(px_038, '.3f')], [format(px_039, '.3f')], [format(px_046, '.3f')], [format(px_047, '.3f')], [format(px_048, '.3f')])
                    print("focal  :", [format(px_031, '.3f')], [format(px_032, '.3f')], [format(px_033, '.3f')], [format(px_040, '.3f')], [format(px_041, '.3f')], [format(px_042, '.3f')], [format(px_049, '.3f')], [format(px_050, '.3f')], [format(px_051, '.3f')])
                    print("focal  :", [format(px_034, '.3f')], [format(px_035, '.3f')], [format(px_036, '.3f')], [format(px_043, '.3f')], [format(px_044, '.3f')], [format(px_045, '.3f')], [format(px_052, '.3f')], [format(px_053, '.3f')], [format(px_054, '.3f')]) 
                    print("periph :", [format(px_055, '.3f')], [format(px_056, '.3f')], [format(px_057, '.3f')], [format(px_064, '.3f')], [format(px_065, '.3f')], [format(px_066, '.3f')], [format(px_073, '.3f')], [format(px_074, '.3f')], [format(px_075, '.3f')])
                    print("periph :", [format(px_058, '.3f')], [format(px_059, '.3f')], [format(px_060, '.3f')], [format(px_067, '.3f')], [format(px_068, '.3f')], [format(px_069, '.3f')], [format(px_076, '.3f')], [format(px_077, '.3f')], [format(px_078, '.3f')])
                    print("periph :", [format(px_061, '.3f')], [format(px_062, '.3f')], [format(px_063, '.3f')], [format(px_070, '.3f')], [format(px_071, '.3f')], [format(px_072, '.3f')], [format(px_079, '.3f')], [format(px_080, '.3f')], [format(px_081, '.3f')])
                except IndexError:
                    print("pv_out: ", "[xxxxxxx]", "[xxxxxxx]", "[xxxxxxx]", "[xxxxxxx]", "[xxxxxxx]", "[xxxxxxx]", "[xxxxxxx]")

                
                signal_dictionary = {          # Create Signal Dictionary and set Pixel in Dictionary equal to Pixel Ref from Image Pixel Matrix
                    'px_001' : [pixel_matrix[y_coord-4][x_coord-4]],         # Section 1 - Top Left
                    'px_002' : [pixel_matrix[y_coord-4][x_coord-3]],
                    'px_003' : [pixel_matrix[y_coord-4][x_coord-2]],
                    'px_004' : [pixel_matrix[y_coord-3][x_coord-4]],
                    'px_005' : [pixel_matrix[y_coord-3][x_coord-3]],
                    'px_006' : [pixel_matrix[y_coord-3][x_coord-2]],
                    'px_007' : [pixel_matrix[y_coord-2][x_coord-4]],
                    'px_008' : [pixel_matrix[y_coord-2][x_coord-3]],
                    'px_009' : [pixel_matrix[y_coord-2][x_coord-2]],

                    'px_010' : [pixel_matrix[y_coord-4][x_coord-1]],         # Section 2 - Top Middle
                    'px_011' : [pixel_matrix[y_coord-4][x_coord+0]],         
                    'px_012' : [pixel_matrix[y_coord-4][x_coord+1]],
                    'px_013' : [pixel_matrix[y_coord-3][x_coord-1]], 
                    'px_014' : [pixel_matrix[y_coord-3][x_coord+0]], 
                    'px_015' : [pixel_matrix[y_coord-3][x_coord+1]],
                    'px_016' : [pixel_matrix[y_coord-2][x_coord-1]], 
                    'px_017' : [pixel_matrix[y_coord-2][x_coord+0]], 
                    'px_018' : [pixel_matrix[y_coord-2][x_coord+1]],

                    'px_019' : [pixel_matrix[y_coord-4][x_coord+2]],         # Section 3 - Top Right
                    'px_020' : [pixel_matrix[y_coord-4][x_coord+3]],
                    'px_021' : [pixel_matrix[y_coord-4][x_coord+4]],
                    'px_022' : [pixel_matrix[y_coord-3][x_coord+2]],
                    'px_023' : [pixel_matrix[y_coord-3][x_coord+3]],
                    'px_024' : [pixel_matrix[y_coord-3][x_coord+4]],
                    'px_025' : [pixel_matrix[y_coord-2][x_coord+2]],         
                    'px_026' : [pixel_matrix[y_coord-2][x_coord+3]],             
                    'px_027' : [pixel_matrix[y_coord-2][x_coord+4]],

                    'px_028' : [pixel_matrix[y_coord-1][x_coord-4]],         # Section 4 - Middle Left
                    'px_029' : [pixel_matrix[y_coord-1][x_coord-3]], 
                    'px_030' : [pixel_matrix[y_coord-1][x_coord-2]], 
                    'px_031' : [pixel_matrix[y_coord+0][x_coord-4]], 
                    'px_032' : [pixel_matrix[y_coord+0][x_coord-3]], 
                    'px_033' : [pixel_matrix[y_coord+0][x_coord-2]], 
                    'px_034' : [pixel_matrix[y_coord+1][x_coord-4]], 
                    'px_035' : [pixel_matrix[y_coord+1][x_coord-3]], 
                    'px_036' : [pixel_matrix[y_coord+1][x_coord-2]],

                    'px_037' : [pixel_matrix[y_coord-1][x_coord-1]],         # Section 5 - Center
                    'px_038' : [pixel_matrix[y_coord-1][x_coord+0]],
                    'px_039' : [pixel_matrix[y_coord-1][x_coord+1]],
                    'px_040' : [pixel_matrix[y_coord+0][x_coord-1]], 
                    'px_041' : [pixel_matrix[y_coord+0][x_coord+0]],         ### Middle of Focal Vision
                    'px_042' : [pixel_matrix[y_coord+0][x_coord+1]],
                    'px_043' : [pixel_matrix[y_coord+1][x_coord-1]], 
                    'px_044' : [pixel_matrix[y_coord+1][x_coord+0]],
                    'px_045' : [pixel_matrix[y_coord+1][x_coord+1]],
                    
                    'px_046' : [pixel_matrix[y_coord-1][x_coord+2]],         # Section 6 - Middle Right
                    'px_047' : [pixel_matrix[y_coord-1][x_coord+3]],
                    'px_048' : [pixel_matrix[y_coord-1][x_coord+4]],
                    'px_049' : [pixel_matrix[y_coord+0][x_coord+2]],
                    'px_050' : [pixel_matrix[y_coord+0][x_coord+3]],
                    'px_051' : [pixel_matrix[y_coord+0][x_coord+4]],
                    'px_052' : [pixel_matrix[y_coord+1][x_coord+2]],         
                    'px_053' : [pixel_matrix[y_coord+1][x_coord+3]],             
                    'px_054' : [pixel_matrix[y_coord+1][x_coord+4]], 

                    'px_055' : [pixel_matrix[y_coord+2][x_coord-4]],         # Section 7 - Bottom Left
                    'px_056' : [pixel_matrix[y_coord+2][x_coord-3]],
                    'px_057' : [pixel_matrix[y_coord+2][x_coord-2]],
                    'px_058' : [pixel_matrix[y_coord+3][x_coord-4]],
                    'px_059' : [pixel_matrix[y_coord+3][x_coord-3]],
                    'px_060' : [pixel_matrix[y_coord+3][x_coord-2]],
                    'px_061' : [pixel_matrix[y_coord+4][x_coord-4]],
                    'px_062' : [pixel_matrix[y_coord+4][x_coord-3]],
                    'px_063' : [pixel_matrix[y_coord+4][x_coord-2]],

                    'px_064' : [pixel_matrix[y_coord+2][x_coord-1]],         # Section 8 - Bottom Middle
                    'px_065' : [pixel_matrix[y_coord+2][x_coord+0]],         
                    'px_066' : [pixel_matrix[y_coord+2][x_coord+1]],
                    'px_067' : [pixel_matrix[y_coord+3][x_coord-1]], 
                    'px_068' : [pixel_matrix[y_coord+3][x_coord+0]], 
                    'px_069' : [pixel_matrix[y_coord+3][x_coord+1]],
                    'px_070' : [pixel_matrix[y_coord+4][x_coord-1]], 
                    'px_071' : [pixel_matrix[y_coord+4][x_coord+0]], 
                    'px_072' : [pixel_matrix[y_coord+4][x_coord+1]],

                    'px_073' : [pixel_matrix[y_coord+2][x_coord+2]],         # Section 9 - Bottom Right
                    'px_074' : [pixel_matrix[y_coord+2][x_coord+3]],
                    'px_075' : [pixel_matrix[y_coord+2][x_coord+4]],
                    'px_076' : [pixel_matrix[y_coord+3][x_coord+2]],
                    'px_077' : [pixel_matrix[y_coord+3][x_coord+3]],
                    'px_078' : [pixel_matrix[y_coord+3][x_coord+4]],
                    'px_079' : [pixel_matrix[y_coord+4][x_coord+2]],         
                    'px_080' : [pixel_matrix[y_coord+4][x_coord+3]],             
                    'px_081' : [pixel_matrix[y_coord+4][x_coord+4]],
                }

                print(signal_dictionary)

                if SHOW_VF_IMAGE == 'ON':
                    # Create figure and axes
                    fig, ax = plt.subplots()
                    # Display the image
                    ax.imshow(img)
                    # Create a Rectangle patch
                    rect = patches.Rectangle((x_coord-4.5, y_coord-4.5), 9, 9, linewidth=1, edgecolor='r', facecolor='none')
                    # Add the patch to the Axes
                    ax.add_patch(rect)
                    plt.pause(PAUSE_TIME)
                    plt.close()

                
                ''' Signal Loop '''
                while signal_dictionary:

                    print("STARTING WHILE LOOP...")
                    if SHOW_VF_IMAGE == 'ON':
                        # Create figure and axes
                        fig, ax = plt.subplots()
                        # Display the image
                        ax.imshow(img)
                        # Create a Rectangle patch
                        rect = patches.Rectangle((x_coord-4.5, y_coord-4.5), 9, 9, linewidth=1, edgecolor='r', facecolor='none')
                        # Add the patch to the Axes
                        ax.add_patch(rect)
                        plt.pause(PAUSE_TIME)
                        plt.close()

                    # If to Shape Identifying Layer, make prediction
                    if 'ss_001' in signal_dictionary.keys():

                        prediction, prediction_value, prediction_certainty, all_values = make_prediction(signal_dictionary)
                        print(f'\n{file_name}')
                        print(f'ACTUAL: {shape} \nPREDICTION: {prediction} - {prediction_certainty}% Certainty \n>>> {prediction_value} in {all_values}\n\n\n')

                        shape_prediction_performance_list.append([shape, prediction, prediction_certainty])

                        signal_dictionary.clear()

                        if SHOW_VF_IMAGE == 'ON':
                            # Create figure and axes
                            fig, ax = plt.subplots()
                            # Display the image
                            ax.imshow(img)
                            # Create a Rectangle patch
                            rect = patches.Rectangle((x_coord-4.5, y_coord-4.5), 9, 9, linewidth=1, edgecolor='r', facecolor='none')
                            # Add the patch to the Axes
                            ax.add_patch(rect)
                            ax.text(2, 3, f'PREDICTON: {prediction}', fontsize=16)
                            plt.pause(PAUSE_TIME*2)
                            plt.close()

                    else:
                    
                        # Run concurrent neuron firings
                        with concurrent.futures.ProcessPoolExecutor() as executor:
                            futures = {executor.submit(generate_signal, neuron) for neuron in signal_dictionary}

                            new_signal_dictionary = {}
                            # update_neuron_dict = {}
                            for neuron in concurrent.futures.as_completed(futures):
                                signal_outputs, post_synaptic_neighbors, neuron_id, signal = neuron.result()

                                if signal is not None:

                                    if neuron_id.startswith("bp"):
                                        bipolar_value_dictionary[neuron_id] = signal_outputs[0]

                                    elif neuron_id.startswith("gn"):
                                        gn_direction_deciding_inputs[neuron_id] = sum(signal_outputs) / len(signal_outputs)

                                    else:
                                        pass

                                    for neighbor, signal_output in zip(post_synaptic_neighbors, signal_outputs):
                                        
                                        if neuron_id in bp_blacklist:
                                            continue

                                        else:
                                            new_signal_dictionary.setdefault(neighbor,[]).append(signal_output)
                                    
                                    # update_neuron_dict[neuron_id] = signal
                                
                                else:
                                    pass

                            # Update signal_dictionary with which neurons to fire next
                            signal_dictionary.clear()
                            signal_dictionary.update(new_signal_dictionary)
                            print("\nUPDATED SIGNAL PATH:", signal_dictionary)

                            # Collect Bipolar Values for Ganglions to use as comparative BP Neighbor Averages
                            if bipolar_value_dictionary:
                                new_ganglion_n_bpv_dictionary = {}
                                for gn in ganglion_neighbors_dictionary:
                                    bpv_list = []
                                    for bp in ganglion_neighbors_dictionary[gn]:
                                        bpv = bipolar_value_dictionary[bp] if bipolar_value_dictionary[bp] else 0
                                        bpv_list.append(bpv)
                                    new_ganglion_n_bpv_dictionary[gn] = bpv_list
                                bipolar_value_dictionary.clear()
                                ganglion_n_bpv_dictionary.update(new_ganglion_n_bpv_dictionary)

                            # Use Ganglion Outputs to determine if Vision Field is moved or stationary
                            elif gn_direction_deciding_inputs:
                                for dd in dd_ganglion_dictionary:
                                    gn_list = []
                                    for gn in dd_ganglion_dictionary[dd]:
                                        gn = gn_direction_deciding_inputs[gn] if gn_direction_deciding_inputs[gn] else 0
                                        gn_list.append(gn)
                                    dd_signal_dictionary[dd] = gn_list
                                print("\n>>>>> dd_signal_dictionary", dd_signal_dictionary)
                                gn_direction_deciding_inputs.clear()

                                ''' Choose Movement Direction or Stay in Place & Make Prediction '''
                                new_x_move, new_y_move = chose_direction(list(dd_signal_dictionary.keys()))
                                dd_signal_dictionary.clear()

                                if new_x_move == 0 and new_y_move == 0:
                                    print("No Move - Making Prediction ... ")
                                    # signal_dictionary maintained
                                    print("SIGNAL DICTIONARY", signal_dictionary)
                                    continue
                                else:
                                    print(f"New_x_move, New_y_move: {new_x_move}x, {new_y_move}y")
                                    # new_signal_dictory cleared so that signal_dictionary may be updated
                                    new_signal_dictionary.clear()

                                    ''' UPDATE MOVEMENT DIRECTION '''
                                    x_move=new_x_move
                                    y_move=new_y_move
                                    x_coord+=x_move
                                    y_coord+=y_move
                                    t+=1

                                    path_dictionary[t] = [x_coord+1, y_coord+1]
                                    print(f"\n\nMOVING FOCUS... NEW TARGET: {x_coord+1}x, {y_coord+1}y")
                                    print("PATH DICTIONARY: ", path_dictionary)

                                    new_signal_dictionary = {                               # Put in NEW Signal Dictionary
                                        'px_001' : [pixel_matrix[y_coord-4][x_coord-4]],         # Section 1 - Top Left
                                        'px_003' : [pixel_matrix[y_coord-4][x_coord-2]],
                                        'px_002' : [pixel_matrix[y_coord-4][x_coord-3]],
                                        'px_004' : [pixel_matrix[y_coord-3][x_coord-4]],
                                        'px_005' : [pixel_matrix[y_coord-3][x_coord-3]],
                                        'px_006' : [pixel_matrix[y_coord-3][x_coord-2]],
                                        'px_007' : [pixel_matrix[y_coord-2][x_coord-4]],
                                        'px_008' : [pixel_matrix[y_coord-2][x_coord-3]],
                                        'px_009' : [pixel_matrix[y_coord-2][x_coord-2]],

                                        'px_010' : [pixel_matrix[y_coord-4][x_coord-1]],         # Section 2 - Top Middle
                                        'px_011' : [pixel_matrix[y_coord-4][x_coord+0]],         
                                        'px_012' : [pixel_matrix[y_coord-4][x_coord+1]],
                                        'px_013' : [pixel_matrix[y_coord-3][x_coord-1]], 
                                        'px_014' : [pixel_matrix[y_coord-3][x_coord+0]], 
                                        'px_015' : [pixel_matrix[y_coord-3][x_coord+1]],
                                        'px_016' : [pixel_matrix[y_coord-2][x_coord-1]], 
                                        'px_017' : [pixel_matrix[y_coord-2][x_coord+0]], 
                                        'px_018' : [pixel_matrix[y_coord-2][x_coord+1]],

                                        'px_019' : [pixel_matrix[y_coord-4][x_coord+2]],         # Section 3 - Top Right
                                        'px_020' : [pixel_matrix[y_coord-4][x_coord+3]],
                                        'px_021' : [pixel_matrix[y_coord-4][x_coord+4]],
                                        'px_022' : [pixel_matrix[y_coord-3][x_coord+2]],
                                        'px_023' : [pixel_matrix[y_coord-3][x_coord+3]],
                                        'px_024' : [pixel_matrix[y_coord-3][x_coord+4]],
                                        'px_025' : [pixel_matrix[y_coord-2][x_coord+2]],         
                                        'px_026' : [pixel_matrix[y_coord-2][x_coord+3]],             
                                        'px_027' : [pixel_matrix[y_coord-2][x_coord+4]],

                                        'px_028' : [pixel_matrix[y_coord-1][x_coord-4]],         # Section 4 - Middle Left
                                        'px_029' : [pixel_matrix[y_coord-1][x_coord-3]], 
                                        'px_030' : [pixel_matrix[y_coord-1][x_coord-2]], 
                                        'px_031' : [pixel_matrix[y_coord+0][x_coord-4]], 
                                        'px_032' : [pixel_matrix[y_coord+0][x_coord-3]], 
                                        'px_033' : [pixel_matrix[y_coord+0][x_coord-2]], 
                                        'px_034' : [pixel_matrix[y_coord+1][x_coord-4]], 
                                        'px_035' : [pixel_matrix[y_coord+1][x_coord-3]], 
                                        'px_036' : [pixel_matrix[y_coord+1][x_coord-2]],

                                        'px_037' : [pixel_matrix[y_coord-1][x_coord-1]],         # Section 5 - Center
                                        'px_038' : [pixel_matrix[y_coord-1][x_coord+0]],
                                        'px_039' : [pixel_matrix[y_coord-1][x_coord+1]],
                                        'px_040' : [pixel_matrix[y_coord+0][x_coord-1]], 
                                        'px_041' : [pixel_matrix[y_coord+0][x_coord+0]],         ### Middle of Focal Vision
                                        'px_042' : [pixel_matrix[y_coord+0][x_coord+1]],
                                        'px_043' : [pixel_matrix[y_coord+1][x_coord-1]], 
                                        'px_044' : [pixel_matrix[y_coord+1][x_coord+0]],
                                        'px_045' : [pixel_matrix[y_coord+1][x_coord+1]],
                                        
                                        'px_046' : [pixel_matrix[y_coord-1][x_coord+2]],         # Section 6 - Middle Right
                                        'px_047' : [pixel_matrix[y_coord-1][x_coord+3]],
                                        'px_048' : [pixel_matrix[y_coord-1][x_coord+4]],
                                        'px_049' : [pixel_matrix[y_coord+0][x_coord+2]],
                                        'px_050' : [pixel_matrix[y_coord+0][x_coord+3]],
                                        'px_051' : [pixel_matrix[y_coord+0][x_coord+4]],
                                        'px_052' : [pixel_matrix[y_coord+1][x_coord+2]],         
                                        'px_053' : [pixel_matrix[y_coord+1][x_coord+3]],             
                                        'px_054' : [pixel_matrix[y_coord+1][x_coord+4]], 

                                        'px_055' : [pixel_matrix[y_coord+2][x_coord-4]],         # Section 7 - Bottom Left
                                        'px_056' : [pixel_matrix[y_coord+2][x_coord-3]],
                                        'px_057' : [pixel_matrix[y_coord+2][x_coord-2]],
                                        'px_058' : [pixel_matrix[y_coord+3][x_coord-4]],
                                        'px_059' : [pixel_matrix[y_coord+3][x_coord-3]],
                                        'px_060' : [pixel_matrix[y_coord+3][x_coord-2]],
                                        'px_061' : [pixel_matrix[y_coord+4][x_coord-4]],
                                        'px_062' : [pixel_matrix[y_coord+4][x_coord-3]],
                                        'px_063' : [pixel_matrix[y_coord+4][x_coord-2]],

                                        'px_064' : [pixel_matrix[y_coord+2][x_coord-1]],         # Section 8 - Bottom Middle
                                        'px_065' : [pixel_matrix[y_coord+2][x_coord+0]],         
                                        'px_066' : [pixel_matrix[y_coord+2][x_coord+1]],
                                        'px_067' : [pixel_matrix[y_coord+3][x_coord-1]], 
                                        'px_068' : [pixel_matrix[y_coord+3][x_coord+0]], 
                                        'px_069' : [pixel_matrix[y_coord+3][x_coord+1]],
                                        'px_070' : [pixel_matrix[y_coord+4][x_coord-1]], 
                                        'px_071' : [pixel_matrix[y_coord+4][x_coord+0]], 
                                        'px_072' : [pixel_matrix[y_coord+4][x_coord+1]],

                                        'px_073' : [pixel_matrix[y_coord+2][x_coord+2]],         # Section 9 - Bottom Right
                                        'px_074' : [pixel_matrix[y_coord+2][x_coord+3]],
                                        'px_075' : [pixel_matrix[y_coord+2][x_coord+4]],
                                        'px_076' : [pixel_matrix[y_coord+3][x_coord+2]],
                                        'px_077' : [pixel_matrix[y_coord+3][x_coord+3]],
                                        'px_078' : [pixel_matrix[y_coord+3][x_coord+4]],
                                        'px_079' : [pixel_matrix[y_coord+4][x_coord+2]],         
                                        'px_080' : [pixel_matrix[y_coord+4][x_coord+3]],             
                                        'px_081' : [pixel_matrix[y_coord+4][x_coord+4]],
                                    }
                                    
                                    signal_dictionary.clear()
                                    signal_dictionary.update(new_signal_dictionary)         # Update 'signal_dictionary' w/ NEW Signal Dictionary
                                    print(f"\n\nUPDATED SIGNAL DICTIONARY: {signal_dictionary}")
                    
                    time.sleep(SLEEP_TIME)      # sleep before beginning next While Loop Iteration


        print(f'MUTATION: {mutation}  -  SHAPE & PREDICTION: {shape_prediction_performance_list}')
        mutation_performance[mutation] = shape_prediction_performance_list

    print(f'MUTATION PERFORMANCE: {mutation_performance}')

    mutation_fitness = {}
    for mutation in mutation_performance:

        all_predictions_list = mutation_performance[mutation]

        score = 0
        for prediction in all_predictions_list:

            if prediction[0] == prediction[1]:

                prediction_certainty = prediction[2]
                score = score + 10 + (prediction_certainty - 33.333)

            else:
                pass

        mutation_fitness[mutation] = round(score, 5)

    print(mutation_fitness)
      


''' Signal Functions '''
# Generate Signal through 1 Neuron
def generate_signal(neuron_from_signal_dictionary):   # nucleus

    neuron = neuron_from_signal_dictionary   # neuron id
    neuron_id = neuron
    signal_input = signal_dictionary[neuron]   # neuron's signal
    # print("Neuron & Signal Input:", neuron_id, signal_input)
    
    excitation = sum(signal_input)
    # print("Neuron & Signal:", neuron_id, excitation)

    if neuron_id.startswith("px") or neuron_id.startswith("pc") or neuron_id.startswith("pr") or neuron_id.startswith("fc") or neuron_id.startswith("bp"):        
        if excitation >= ACTIVATION_THRESHOLD:
            post_synaptic_neighbors = starting_post_synaptic_neighbors_dictionary[neuron][0]
            synapse_values = starting_post_synaptic_neighbors_dictionary[neuron][1]
            signal_outputs = [round(synapse_value * excitation, 5) for synapse_value in synapse_values]   # multiply excitation by array of synapse_values

        else:
            signal_outputs = None
            post_synaptic_neighbors = None
            neuron_id = None
            excitation = None

    elif neuron_id.startswith("gn"):
        bpv = excitation
        # bpv = excitation
        avg_n_bpv = max(sum(ganglion_n_bpv_dictionary[neuron_id]) / len(ganglion_n_bpv_dictionary[neuron_id]), 0.001)
        rate = bpv / avg_n_bpv

        post_synaptic_neighbors = starting_post_synaptic_neighbors_dictionary[neuron][0]
        synapse_values = starting_post_synaptic_neighbors_dictionary[neuron][1]
        signal_outputs = [round(synapse_value * rate, 5) for synapse_value in synapse_values]   # multiply excitation by array of synapse_values
        # print(">>>>>>", neuron_id, "\n",
        #         "signal", signal_input, bpv, "\n",
        #         "n_avg ", avg_n_bpv, ganglion_n_bpv_dictionary[neuron_id], "\n",
        #         "rate  ", rate, "-", signal_outputs, "\n", 
        #         "output", ganglion_n_bpv_dictionary[neuron_id], "\n")
  
    else:
        if excitation >= ACTIVATION_THRESHOLD:
            post_synaptic_neighbors = starting_post_synaptic_neighbors_dictionary[neuron][0]
            synapse_values = starting_post_synaptic_neighbors_dictionary[neuron][1]
            # signal_outputs = [synapse_value * excitation for synapse_value in synapse_values]   # multiply excitation by array of synapse_values
            signal_outputs = synapse_values
        else:
            signal_outputs = None
            post_synaptic_neighbors = None
            neuron_id = None
            excitation = None

    return signal_outputs, post_synaptic_neighbors, neuron_id, excitation

# Use the Signal to determine which direction to move
def chose_direction(neurons_firing):

    direction_neuron = None
    highest_signal = 0

    for neuron in neurons_firing:
        
        signal_sum = sum(dd_signal_dictionary[neuron])
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

def make_prediction(ssn_signal_dictionary):

    ssn_sums = {}
    for ssn_id in ssn_signal_dictionary:
        ssn_sums[ssn_id] = sum(ssn_signal_dictionary[ssn_id])

    triangle_value = 0
    circle_value = 0
    square_value = 0
    for ssn_id in ssn_sums:
        if ssn_id in ('ss_001', 'ss_002', 'ss_003', 'ss_004'):
            triangle_value+=ssn_sums[ssn_id]
        elif ssn_id in ('ss_005', 'ss_006', 'ss_007', 'ss_008'):
            circle_value+=ssn_sums[ssn_id]
        else:
            square_value+=ssn_sums[ssn_id]

    shape_value_list = [triangle_value, circle_value, square_value]
    shape_list = ['triangle', 'circle', 'square']

    prediction_value = max(triangle_value, circle_value, square_value)
    index = shape_value_list.index(prediction_value)
    prediction = shape_list[index]
    all_values = [triangle_value, circle_value, square_value]

    prediction_certainty = round(prediction_value / sum(all_values), 5)*100

    return prediction, prediction_value, prediction_certainty, all_values



''' RUN ''' 
if __name__ == "__main__":

    main()
    