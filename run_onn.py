

import time
import ast
from PIL import Image
import concurrent.futures


X_COORD = 5             # For reading index of Image Pixel Matrix (List of lists)
Y_COORD = 5             # (!) At sampling COORD is subtracted by 1 ('-1') due to referencing 0-th Indexed List
EPOCHS = 6              # Number of times to run training


''' Load the Neural Network '''
with open("onn_map.json", "r") as file:
    starting_post_synaptic_neighbors_dictionary = file.read()
# Convert to Python Dictionary
starting_post_synaptic_neighbors_dictionary = ast.literal_eval(starting_post_synaptic_neighbors_dictionary)


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


''' Signal Functions '''
# Generate Signal through 1 Neuron
def generate_signal(neuron_from_signal_dictionary):   # nucleus

    neuron = neuron_from_signal_dictionary   # neuron id
    neuron_id = neuron
    signal_input = signal_dictionary[neuron]   # neuron's signal
    print("Neuron & Signal Input:", neuron_id, signal_input)
    
    signal = sum(signal_input)
    print("Neuron & Signal:", neuron_id, signal)

    if signal > 0:
        
        post_synaptic_neighbors = starting_post_synaptic_neighbors_dictionary[neuron][0]
        synapse_values = starting_post_synaptic_neighbors_dictionary[neuron][1]
        signal_outputs = [synapse_value * signal for synapse_value in synapse_values]   # multiply signal by array of synapse_values

    else:

        signal_outputs = None
        post_synaptic_neighbors = None
        neuron_id = None
        signal = None
    
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
    x_coord = X_COORD -1
    y_coord = Y_COORD -1

    for _ in range(EPOCHS):

        ''' Signal Loop '''
        while signal_dictionary:

            if any(key in direction_lookup for key in signal_dictionary):
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

                        if signal is not None:
                            for neighbor, signal_output in zip(post_synaptic_neighbors, signal_outputs):
                                new_signal_dict.setdefault(neighbor,[]).append(signal_output)
                            update_neuron_dict[neuron_id] = signal
                        
                        else:
                            pass

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
        print(f"\n\nMOVING FOCUS... NEW TARGET: {x_coord+1}x, {y_coord+1}y")

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
        signal_dictionary.update(new_signal_dictionary)         # Update 'signal_dictionary' w/ NEW Signal Dictionary
        
        print(f"\n\nNEW SIGNAL DICTIONARY: {signal_dictionary}")

        t+=1





""" RUN """
if __name__ == "__main__":

    t = 0
    x_coord = X_COORD -1        # For reading index of Image Pixel Matrix (List of lists)
    y_coord = Y_COORD -1        # (!) At sampling COORD is subtracted by 1 ('-1') due to referencing 0-th Indexed List


    # Set Field of Vision Pixel Inputs
    ''' px_001 references the 1st Pixel of the Field of Vision NOT the image !!! '''

    pixel_input_list = []
    p_num = 1
    for p in range(81):
        pixel = str(p_num)
        pixel = pixel.zfill(3)
        pixel_input_list.append(f'px_{pixel}')
        p_num+=1

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
    