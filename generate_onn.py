

import random
from random import randrange
import numpy as np
# random.seed(42)
# np.random.seed(42)
print("\n")


''' NOTES

    The final product of this script is to generate the starting "brain map" or Neural Network for training. 
    This "brain map" is a dictionary (or JSON) of every Neuron's post-synaptic connections.

    ie. Ganglion 'gn_026' connects to Occipital Entry Neurons 'oc_031', 'oc_032', & 'oc_033' w/ positive synapse values
            of '0.832', '0.757', & '0.945' respectively.

        'gn_026': [
            ['oc_031', 'oc_032', 'oc_033'],
            [0.832, 0.757, 0.945]
        ],

        As the signal passes through the Network...
        
        This is done via run_onn.py.

    Most code needs to be reformatted for flexibitliy & efficiency. I'm moving very quickly for testing purposes,
    as my major goal is to test the translations of fundamental Neuroscience principles -> AI principles.

'''



""" HELPERS """

pos_or_neg_list = [-1, 1]
def random_pos_or_neg():
    pos_or_neg = random.choice(pos_or_neg_list)
    return pos_or_neg

def gen_syn_list(num):
    pos_or_neg = random_pos_or_neg()
    syn_list = []
    for _ in range(num):
        syn_list.append((pos_or_neg)*np.round(np.random.uniform(0.70, 0.99), 3))
    return syn_list

""" DEFINE VISUAL SPACE """

FOCAL_VISION_FIELD_WIDTH = 3            # Most dimensions are in Pixels  (ie. Focal Vision Width of 3 = 3 Pixels Wide)
FOCAL_VISION_FIELD_HEIGHT = 3           
TOTAL_VISION_FIELD_WIDTH = 9            # Includes Focal Vision Field
TOTAL_VISION_FIELD_HEIGHT = 9           # (!) NEEDS Update to accomodate differently sized Focal and Peripheral Sections
PERIPHERAL_VISION_SECTIONS = 8          # Number of Periheral Vision Sections (8 Peripheral + 1 Focal = 9 Basic Directions)
                                        # Sections are numbered 1 to 9 from top left to bottom right
# Shorthand
PERIPHERAL_VISION_SPACE = ((TOTAL_VISION_FIELD_WIDTH*TOTAL_VISION_FIELD_HEIGHT)-(FOCAL_VISION_FIELD_WIDTH*FOCAL_VISION_FIELD_HEIGHT))

FOCAL_CONES_PER_PIXEL = 1               # ex. 2 Focal Cones per 1 Focal Pixel
RODS_PER_PIXEL = 1
FOCAL_BIPOLAR_PER_PIXEL = 1                  # Each Focal Cone Gets 1 Bipolar Cell & 1 Ganglion
PERIPHERAL_BIPOLAR_PER_PIXEL = 1*1/9         # Each Section (9px) gets 2 Bipolar Cells & 1 Ganglion
FOCAL_GANGLION_PER_PIXEL = 1
PERIPHERAL_GANGLION_PER_PIXEL = 1*1/9

# Dimensions are per Section(ish)
FOCAL_OCCIPITAL_PER_SECTION = 9              # There is no Spacial need of Occipital per Pixel (multiple of 3 makes math easier)
PERIPHERAL_OCCIPITAL_PER_SECTION = 3
DEEP_OCCIPITAL_PER_ENTRY = 3                 # (!) Per "Occipital Entry Neuron"


# Pixel Input
visual_field_dictionary = {}
section_num = 1
pixel_num = 1
for _ in range(PERIPHERAL_VISION_SECTIONS+1):
    section = str(section_num)
    section = section.zfill(3)

    pixel_list = []
    for _ in range(FOCAL_VISION_FIELD_WIDTH*FOCAL_VISION_FIELD_HEIGHT):
        pixel = str(pixel_num)
        pixel = pixel.zfill(3)
        pixel_list.append(f'px_{pixel}')
        pixel_num+=1

    visual_field_dictionary[f'vf_{section}'] = [pixel_list]
    section_num+=1

# Rods & Cones
focal_cone_num = 1
rod_num = 1
for vf_section in visual_field_dictionary:
    
    if 'vf_005' in vf_section:
        focal_cone_list = []
        for _ in range(FOCAL_CONES_PER_PIXEL*FOCAL_VISION_FIELD_WIDTH*FOCAL_VISION_FIELD_HEIGHT):
            cone = str(focal_cone_num)
            cone = cone.zfill(3)
            focal_cone_list.append(f'fc_{cone}')
            focal_cone_num+=1
        visual_field_dictionary[vf_section].append(focal_cone_list)

    else:
        rod_list = []
        for _ in range(int(RODS_PER_PIXEL*PERIPHERAL_VISION_SPACE/PERIPHERAL_VISION_SECTIONS)):
            rod = str(rod_num)
            rod = rod.zfill(3)
            rod_list.append(f'pr_{rod}')
            rod_num+=1
        visual_field_dictionary[vf_section].append(rod_list)

# Bipolar Cells
bipolar_num = 1
for vf_section in visual_field_dictionary:

    if 'vf_005' in vf_section:
        focal_bipolar_list = []
        for _ in range(FOCAL_BIPOLAR_PER_PIXEL*FOCAL_VISION_FIELD_WIDTH*FOCAL_VISION_FIELD_HEIGHT):
            bipolar = str(bipolar_num)
            bipolar = bipolar.zfill(3)
            focal_bipolar_list.append(f'bp_{bipolar}')
            bipolar_num+=1
        visual_field_dictionary[vf_section].append(focal_bipolar_list)

    else:
        peripheral_bipolar_list = []
        for _ in range(int(PERIPHERAL_BIPOLAR_PER_PIXEL*PERIPHERAL_VISION_SPACE/PERIPHERAL_VISION_SECTIONS)):
            bipolar = str(bipolar_num)
            bipolar = bipolar.zfill(3)
            peripheral_bipolar_list.append(f'bp_{bipolar}')
            bipolar_num+=1
        visual_field_dictionary[vf_section].append(peripheral_bipolar_list)

# Ganglion Neurons
ganglion_num = 1
for vf_section in visual_field_dictionary:

    if 'vf_005' in vf_section:
        focal_ganglion_list = []
        for _ in range(FOCAL_GANGLION_PER_PIXEL*FOCAL_VISION_FIELD_WIDTH*FOCAL_VISION_FIELD_HEIGHT):
            ganglion = str(ganglion_num)
            ganglion = ganglion.zfill(3)
            focal_ganglion_list.append(f'gn_{ganglion}')
            ganglion_num+=1
        visual_field_dictionary[vf_section].append(focal_ganglion_list)

    else:
        peripheral_ganglion_list = []
        for _ in range(int(PERIPHERAL_GANGLION_PER_PIXEL*PERIPHERAL_VISION_SPACE/PERIPHERAL_VISION_SECTIONS)):
            ganglion = str(ganglion_num)
            ganglion = ganglion.zfill(3)
            peripheral_ganglion_list.append(f'gn_{ganglion}')
            ganglion_num+=1
        visual_field_dictionary[vf_section].append(peripheral_ganglion_list)

# Occipital Neurons - Entry
occipital_num = 1
for vf_section in visual_field_dictionary:

    if 'vf_005' in vf_section:
        focal_occipital_list = []
        for _ in range(FOCAL_OCCIPITAL_PER_SECTION):
            occipital = str(occipital_num)
            occipital = occipital.zfill(3)
            focal_occipital_list.append(f'oc_{occipital}')
            occipital_num+=1
        visual_field_dictionary[vf_section].append(focal_occipital_list)

    else:
        peripheral_occipital_list = []
        for _ in range(int(PERIPHERAL_OCCIPITAL_PER_SECTION)):
            occipital = str(occipital_num)
            occipital = occipital.zfill(3)
            peripheral_occipital_list.append(f'oc_{occipital}')
            occipital_num+=1
        visual_field_dictionary[vf_section].append(peripheral_occipital_list)

# Occipital Neurons - Deep
deep_occipital_num = 1
for vf_section in visual_field_dictionary:

    if 'vf_005' in vf_section:
        deep_occipital_list = []
        for _ in range(DEEP_OCCIPITAL_PER_ENTRY*FOCAL_OCCIPITAL_PER_SECTION):
            occipital = str(deep_occipital_num)
            occipital = occipital.zfill(3)
            deep_occipital_list.append(f'on_{occipital}')
            deep_occipital_num+=1
        visual_field_dictionary[vf_section].append(deep_occipital_list)

    else:
        deep_occipital_list = []
        for _ in range(int(DEEP_OCCIPITAL_PER_ENTRY*PERIPHERAL_OCCIPITAL_PER_SECTION)):
            occipital = str(deep_occipital_num)
            occipital = occipital.zfill(3)
            deep_occipital_list.append(f'on_{occipital}')
            deep_occipital_num+=1
        visual_field_dictionary[vf_section].append(deep_occipital_list)

# Direction Deciding Neurons - Feedback to Visual Field Motor Function
direction_deciding_num = 1
for vf_section in visual_field_dictionary:

    ddn = str(direction_deciding_num)
    ddn = ddn.zfill(3)
    visual_field_dictionary[vf_section].append([f'dd_{ddn}'])
    direction_deciding_num+=1

# print(visual_field_dictionary)



""" GENERATE SPACIAL MAP """

oc_spacial_map = {}                 # Spacial Layer Map as Dictionary of List of Lists
on_spacial_map = {} 
oc_section_col_count = 1
oc_section_row_count = 1
on_section_col_count = 1
on_section_row_count = 1
oc_num = 1
on_num = 1
oc_section_row = []
on_section_row = []
oc_row_n = []
oc_row_n_1 = []
oc_row_n_2 = []
on_row_n = []
on_row_n_1 = []
on_row_n_2 = []
for vf_section in visual_field_dictionary:

    ''' Occipital Entry Layer '''     
    if oc_section_row_count == 2 and oc_section_col_count == 2:
        oc_row_n.extend(visual_field_dictionary[vf_section][4][0:3])
        oc_row_n_1.extend(visual_field_dictionary[vf_section][4][3:6])
        oc_row_n_2.extend(visual_field_dictionary[vf_section][4][6:9])
        oc_section_col_count+=1

    elif oc_section_col_count < 3:
        oc_row_n.extend(visual_field_dictionary[vf_section][4][0:1])
        oc_row_n_1.extend(visual_field_dictionary[vf_section][4][1:2])
        oc_row_n_2.extend(visual_field_dictionary[vf_section][4][2:3])
        oc_section_col_count+=1

    else:
        oc_row_n.extend(visual_field_dictionary[vf_section][4][0:1])
        oc_row_n_1.extend(visual_field_dictionary[vf_section][4][1:2])
        oc_row_n_2.extend(visual_field_dictionary[vf_section][4][2:3])
        oc_spacial_map[f'oc_row_{oc_num}'] = oc_row_n
        oc_spacial_map[f'oc_row_{oc_num+1}'] = oc_row_n_1
        oc_spacial_map[f'oc_row_{oc_num+2}'] = oc_row_n_2
        oc_num+=3
        oc_row_n = []
        oc_row_n_1 = []
        oc_row_n_2 = []
        oc_section_col_count = 1
        oc_section_row_count+=1

        
    ''' Occipital Deep Layer '''   
    if on_section_row_count == 2 and on_section_col_count == 2:
        on_row_n.extend(visual_field_dictionary[vf_section][5][0:9])
        on_row_n_1.extend(visual_field_dictionary[vf_section][5][9:18])
        on_row_n_2.extend(visual_field_dictionary[vf_section][5][18:27])
        on_section_col_count+=1

    elif on_section_col_count < 3:
        on_row_n.extend(visual_field_dictionary[vf_section][5][0:3])
        on_row_n_1.extend(visual_field_dictionary[vf_section][5][3:6])
        on_row_n_2.extend(visual_field_dictionary[vf_section][5][6:9])
        on_section_col_count+=1

    else:
        on_row_n.extend(visual_field_dictionary[vf_section][5][0:3])
        on_row_n_1.extend(visual_field_dictionary[vf_section][5][3:6])
        on_row_n_2.extend(visual_field_dictionary[vf_section][5][6:9])
        on_spacial_map[f'on_row_{on_num}'] = on_row_n
        on_spacial_map[f'on_row_{on_num+1}'] = on_row_n_1
        on_spacial_map[f'on_row_{on_num+2}'] = on_row_n_2
        on_num+=3
        on_row_n = []
        on_row_n_1 = []
        on_row_n_2 = []
        on_section_col_count = 1
        on_section_row_count+=1

# print(oc_spacial_map)
# print(on_spacial_map)

on_spacial_map = {          # Edited to include "None" for blank spaces. Reformat. Edited in this way to perserve time for testing.
    'on_row_1': ['on_001', 'on_002', 'on_003',   None,   'on_010',    None,    None,   'on_011',    None,    None,   'on_012',   None,   'on_019', 'on_020', 'on_021'],
    'on_row_2': ['on_004', 'on_005', 'on_006',   None,   'on_013',    None,    None,   'on_014',    None,    None,   'on_015',   None,   'on_022', 'on_023', 'on_024'],
    'on_row_3': ['on_007', 'on_008', 'on_009',   None,   'on_016',    None,    None,   'on_017',    None,    None,   'on_018',   None,   'on_025', 'on_026', 'on_027'],
    'on_row_4': ['on_028', 'on_029', 'on_030', 'on_037', 'on_038', 'on_039', 'on_040', 'on_041', 'on_042', 'on_043', 'on_044', 'on_045', 'on_064', 'on_065', 'on_066'],
    'on_row_5': ['on_031', 'on_032', 'on_033', 'on_046', 'on_047', 'on_048', 'on_049', 'on_050', 'on_051', 'on_052', 'on_053', 'on_054', 'on_067', 'on_068', 'on_069'],
    'on_row_6': ['on_034', 'on_035', 'on_036', 'on_055', 'on_056', 'on_057', 'on_058', 'on_059', 'on_060', 'on_061', 'on_062', 'on_063', 'on_070', 'on_071', 'on_072'],
    'on_row_7': ['on_073', 'on_074', 'on_075',   None,   'on_082',    None,    None,   'on_083',    None,    None,   'on_084',   None,   'on_091', 'on_092', 'on_093'],
    'on_row_8': ['on_076', 'on_077', 'on_078',   None,   'on_085',    None,    None,   'on_086',    None,    None,   'on_087',   None,   'on_094', 'on_095', 'on_096'],
    'on_row_9': ['on_079', 'on_080', 'on_081',   None,   'on_088',    None,    None,   'on_089',    None,    None,   'on_090',   None,   'on_097', 'on_098', 'on_099']
}



""" GENEARATE POST-SYNAPTIC CONNECTIONS """

onn_map = {}
vf_index = 0
circular_list_of_peripheral_sections = ['vf_001', 'vf_002', 'vf_003', 'vf_006', 'vf_009', 'vf_008', 'vf_007', 'vf_004', 'vf_001', 'vf_002', 'vf_003', 'vf_006', 'vf_009', 'vf_008', 'vf_007', 'vf_004']
for vf_section in visual_field_dictionary:
    
    if 'vf_005' in vf_section:                          # If in Focal Vision
        
        fc_num = 0
        pixel_connection_list = []
        for pixel in visual_field_dictionary[vf_section][0]:           # Define Pixel-Focal Cone Relationship
            onn_map[pixel] = [[visual_field_dictionary[vf_section][1][fc_num]],
                                [1]]                             # "1"s just to pass on signal from Pixel
            fc_num+=1
        
        bp_connection_num = 0
        for cone in visual_field_dictionary[vf_section][1]:             # Define Focal Cone-Bipolar Relationship
            onn_map[cone] = [[visual_field_dictionary[vf_section][2][bp_connection_num]],
                                [1]]
            bp_connection_num+=1

        gn_connection_num = 0
        for bipolar in visual_field_dictionary[vf_section][2]:        # Define Focal Bipolar-Ganglion Relationship
            
            onn_map[bipolar] = [[visual_field_dictionary[vf_section][3][gn_connection_num]],
                                [1]]
            gn_connection_num+=1

        ganglion_count = 0
        occipital_count = 0
        for ganglion in visual_field_dictionary[vf_section][3]:

            if (ganglion_count % 2) == 0:
                onn_map[ganglion] = [[visual_field_dictionary[vf_section][4][occipital_count]],
                                        [0.500]]
                occipital_count-=1
            else:
                onn_map[ganglion] = [[visual_field_dictionary[vf_section][4][occipital_count]],
                                        [0.500]]
            ganglion_count+=1
            occipital_count+=1

        row = 1
        col = 1
        on_row = 4
        on_col = 4
        for oc in visual_field_dictionary[vf_section][4]:       # Use Spacial Map to map OC to ON based on spacial dimensions
            if col < 3:
                on_row_n_minus_2 = list(filter(None, on_spacial_map[f'on_row_{on_row-2}'][on_col-2:on_col+3]))
                on_row_n_minus_1 = list(filter(None, on_spacial_map[f'on_row_{on_row-1}'][on_col-2:on_col+3]))
                on_row_n = on_spacial_map[f'on_row_{on_row}'][on_col-2:on_col+3]
                on_row_n_plus_1 = list(filter(None, on_spacial_map[f'on_row_{on_row+1}'][on_col-2:on_col+3]))
                on_row_n_plus_2 = list(filter(None, on_spacial_map[f'on_row_{on_row+2}'][on_col-2:on_col+3]))
                full_list = on_row_n_minus_2 + on_row_n_minus_1 + on_row_n + on_row_n_plus_1 + on_row_n_plus_2
                onn_map[oc] = [full_list,
                                gen_syn_list(len(full_list))]
                on_col+=3
                col+=1
            else:
                on_row_n_minus_2 = list(filter(None, on_spacial_map[f'on_row_{on_row-2}'][on_col-2:on_col+3]))
                on_row_n_minus_1 = list(filter(None, on_spacial_map[f'on_row_{on_row-1}'][on_col-2:on_col+3]))
                on_row_n = on_spacial_map[f'on_row_{on_row}'][on_col-2:on_col+3]
                on_row_n_plus_1 = list(filter(None, on_spacial_map[f'on_row_{on_row+1}'][on_col-2:on_col+3]))
                on_row_n_plus_2 = list(filter(None, on_spacial_map[f'on_row_{on_row+2}'][on_col-2:on_col+3]))
                full_list = on_row_n_minus_2 + on_row_n_minus_1 + on_row_n + on_row_n_plus_1 + on_row_n_plus_2
                onn_map[oc] = [full_list,
                                gen_syn_list(len(full_list))]
                col = 1         # reset column index
                on_row+=1       # go to next row
                on_col = 4

    else:                                                             # Else in Peripheral Vision
        pr_num = 0
        for pixel in visual_field_dictionary[vf_section][0]:          # Define Pixel-Rod Relationship
            onn_map[pixel] = [[visual_field_dictionary[vf_section][1][pr_num]],
                                [1]]
            pr_num+=1

        for rod in visual_field_dictionary[vf_section][1]:            # Define Rod-Bipolar Relationship
            # Connect each Rod to each Bipolar Cell
            onn_map[rod] = [visual_field_dictionary[vf_section][2],
                            [1]]

        gn_connection_num = 0
        for bipolar in visual_field_dictionary[vf_section][2]:            # Define Bipolar-Ganglion Relationship       
            
            # Connect each Bipolar Cell to Ganglion
            onn_map[bipolar] = [visual_field_dictionary[vf_section][3],
                                [1]]
            gn_connection_num+=1

        for ganglion in visual_field_dictionary[vf_section][3]:
            onn_map[ganglion] = [visual_field_dictionary[vf_section][4],
                                    [0.500, 0.500, 0.500]]
        
        oc_num = 1
        for oc in visual_field_dictionary[vf_section][4]:
            if oc_num == 1:                                                      # counter clockwise peripheral vf section
                cc_vf_section = circular_list_of_peripheral_sections[vf_index-1]
                onn_map[oc] = [visual_field_dictionary[cc_vf_section][5],
                                gen_syn_list(12)]
                oc_num+=1
            elif oc_num == 2:                                                    # this vf section
                onn_map[oc] = [visual_field_dictionary[vf_section][5],
                                gen_syn_list(12)]
                oc_num+=1
            else:                                                                # clockwise peripheral vf section
                cc_vf_section = circular_list_of_peripheral_sections[vf_index+1]
                onn_map[oc] = [visual_field_dictionary[cc_vf_section][5],
                                gen_syn_list(12)]
                oc_num = 1
        vf_index+=1

    for on in  visual_field_dictionary[vf_section][5]:                      # Connect all Deep Occipital Neurons to
        onn_map[on] = [visual_field_dictionary[vf_section][6],              # their Section's Direction Deciding Neuron
                        gen_syn_list(1)]


print(onn_map)
