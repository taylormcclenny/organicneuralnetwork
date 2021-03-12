

import random
from random import randrange
import numpy as np
# random.seed(42)
# np.random.seed(42)
print("\n")

""" DEFINE VISUAL SPACE """

# Dimensions are in Pixels
FOCAL_VISION_FIELD_WIDTH = 3            # ie. Focal Vision Width of 3 = 3 Pixels Wide
FOCAL_VISION_FIELD_HEIGHT = 3           
TOTAL_VISION_FIELD_WIDTH = 9            # Includes Focal Vision Field
TOTAL_VISION_FIELD_HEIGHT = 9           # (!) NEEDS Update to accomodate differently sized Focal and Peripheral Sections
PERIPHERAL_VISION_SECTIONS = 8          # Number of Periheral Vision Sections (8 Peripheral + 1 Focal = 9 Basic Directions)
                                        # Sections are numbered 1 to 9 from top left to bottom right

# Shorthand
PERIPHERAL_VISION_SPACE = ((TOTAL_VISION_FIELD_WIDTH*TOTAL_VISION_FIELD_HEIGHT)-(FOCAL_VISION_FIELD_WIDTH*FOCAL_VISION_FIELD_HEIGHT))

FOCAL_CONES_PER_PIXEL = 1               # ex. 1 Focal Cones per 1 Focal Pixel
PERIPHERAL_CONES_PER_PIXEL = 1               # ex. 1 Peripheral Cones per 1 Peripheral Pixel
PERIPHERAL_CONES = 16
RODS_PER_PIXEL = 1
FOCAL_BIPOLAR_PER_PIXEL = 1                  # ex. Each Focal Cone Gets 1 Bipolar Cell & 1 Ganglion
PERIPHERAL_BIPOLAR_PER_PIXEL = 1*1/9         # ex. Each Section (9px) gets 2 Bipolar Cells & 1 Ganglion
FOCAL_GANGLION_PER_PIXEL = 1
PERIPHERAL_GANGLION_PER_PIXEL = 1*1/9

BIPOLAR_PER_ROD = 1/9
BIPOLOAR_PER_CONE = 1

# Dimensions are per Section(ish)
FOCAL_OCCIPITAL_PER_SECTION = 36             # 9 Focal Ganglion x 2 Excitatory Occ. x 2 Inhibitory Occ.
PERIPHERAL_OCCIPITAL_PER_SECTION = 4         # 8 Peripheral Ganglion x 2 Excitatory Occ. x 2 Inhibitory Occ.
NUM_OF_SHAPE_SPECIFIC_CLASSES = 3
STARTING_SHAPE_SPECIFIC_NEURONS_PER_CLASS = 4


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
peripheral_cone_num = 1
rod_num = 1
for vf_section in visual_field_dictionary:
    
    if 'vf_005' in vf_section:                  # Focal Vision Section of "Vision Field"
        focal_cone_list = []
        for _ in range(FOCAL_CONES_PER_PIXEL*FOCAL_VISION_FIELD_WIDTH*FOCAL_VISION_FIELD_HEIGHT):
            cone = str(focal_cone_num)
            cone = cone.zfill(3)
            focal_cone_list.append(f'fc_{cone}')
            focal_cone_num+=1
        visual_field_dictionary[vf_section].append(focal_cone_list)

    else:                                       # for all other (non-Focal) Sections
        photoreceptor_list = []
        # for adding Rods
        for _ in range(int(RODS_PER_PIXEL*PERIPHERAL_VISION_SPACE/PERIPHERAL_VISION_SECTIONS)):
            rod = str(rod_num)
            rod = rod.zfill(3)
            photoreceptor_list.append(f'pr_{rod}')
            rod_num+=1

        # for adding Peripheral Cone to corner Sections
        if vf_section == 'vf_001' or vf_section == 'vf_003' or vf_section == 'vf_007' or vf_section == 'vf_009':      
            p_cone = str(peripheral_cone_num)
            p_cone = p_cone.zfill(3)
            photoreceptor_list.append(f'pc_{p_cone}')
            peripheral_cone_num+=1
        
        # for adding Peripheral Cone to middle Sections
        elif vf_section == 'vf_002' or vf_section == 'vf_004' or vf_section == 'vf_006' or vf_section == 'vf_008':
            peripheral_cone_list = []
            for _ in range(int((PERIPHERAL_CONES/PERIPHERAL_VISION_SECTIONS*2)-1)):
                p_cone = str(peripheral_cone_num)
                p_cone = p_cone.zfill(3)
                photoreceptor_list.append(f'pc_{p_cone}')
                peripheral_cone_num+=1

        visual_field_dictionary[vf_section].append(photoreceptor_list)

# Bipolar Cells
bipolar_num = 1
for vf_section in visual_field_dictionary:

    if 'vf_005' in vf_section:                  # Focal Vision Section of "Vision Field"
        focal_bipolar_list = []
        for _ in range(FOCAL_BIPOLAR_PER_PIXEL*FOCAL_VISION_FIELD_WIDTH*FOCAL_VISION_FIELD_HEIGHT):
            bipolar = str(bipolar_num)
            bipolar = bipolar.zfill(3)
            focal_bipolar_list.append(f'bp_{bipolar}')
            bipolar_num+=1
        visual_field_dictionary[vf_section].append(focal_bipolar_list)

    else:                                       # for Peripheral Sections
        peripheral_bipolar_list = []

        # for adding Peripheral Cone Bipolars to corner Sections
        if vf_section == 'vf_001' or vf_section == 'vf_003' or vf_section == 'vf_007' or vf_section == 'vf_009':      
            bipolar = str(bipolar_num)
            bipolar = bipolar.zfill(3)
            peripheral_bipolar_list.append(f'bp_{bipolar}')
            bipolar_num+=1
        
        # for adding Peripheral Cone Bipolars to middle Sections
        elif vf_section == 'vf_002' or vf_section == 'vf_004' or vf_section == 'vf_006' or vf_section == 'vf_008':
            peripheral_cone_list = []
            for _ in range(int((PERIPHERAL_CONES/PERIPHERAL_VISION_SECTIONS*2)-1)):
                bipolar = str(bipolar_num)
                bipolar = bipolar.zfill(3)
                peripheral_bipolar_list.append(f'bp_{bipolar}')
                bipolar_num+=1

        for _ in range(int(PERIPHERAL_BIPOLAR_PER_PIXEL*PERIPHERAL_VISION_SPACE/PERIPHERAL_VISION_SECTIONS)):
            bipolar = str(bipolar_num)
            bipolar = bipolar.zfill(3)
            peripheral_bipolar_list.append(f'bp_{bipolar}')
            bipolar_num+=1

        visual_field_dictionary[vf_section].append(peripheral_bipolar_list)

# Ganglion Neurons
ganglion_num = 1
for vf_section in visual_field_dictionary:

    if 'vf_005' in vf_section:                  # Focal Vision Section of "Vision Field"
        focal_ganglion_list = []
        for _ in range(FOCAL_GANGLION_PER_PIXEL*FOCAL_VISION_FIELD_WIDTH*FOCAL_VISION_FIELD_HEIGHT):
            ganglion = str(ganglion_num)
            ganglion = ganglion.zfill(3)
            focal_ganglion_list.append(f'gn_{ganglion}')
            ganglion_num+=1
        visual_field_dictionary[vf_section].append(focal_ganglion_list)

    else:                                       # for all other (non-Focal) Sections
        peripheral_ganglion_list = []
        for _ in range(int(PERIPHERAL_GANGLION_PER_PIXEL*PERIPHERAL_VISION_SPACE/PERIPHERAL_VISION_SECTIONS)):
            ganglion = str(ganglion_num)
            ganglion = ganglion.zfill(3)
            peripheral_ganglion_list.append(f'gn_{ganglion}')
            ganglion_num+=1
        visual_field_dictionary[vf_section].append(peripheral_ganglion_list)

# Occipital Entry Neurons - Also Direction Deciding Neurons, Feedback to Visual Field Motor Function
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

# Shape Specific Neurons
shape_specific_class_num = 1
full_ssn_list = []                                              # For referencing ease when connecting all Occ. to all SSN
ssn_num = 1
for _ in range(NUM_OF_SHAPE_SPECIFIC_CLASSES):
    section = str(shape_specific_class_num)
    section = section.zfill(3)

    ssn_list = []
    for _ in range(STARTING_SHAPE_SPECIFIC_NEURONS_PER_CLASS):
        ssn = str(ssn_num)
        ssn = ssn.zfill(3)
        ssn_list.append(f'ss_{ssn}')
        full_ssn_list.append(f'ss_{ssn}')
        ssn_num+=1

    visual_field_dictionary[f'si_{section}'] = [ssn_list]
    shape_specific_class_num+=1

# print(visual_field_dictionary)
# print(full_ssn_list)



""" GENEARATE POST-SYNAPTIC CONNECTIONS """

onn_map = {}
vf_index = 0
for vf_section in visual_field_dictionary:
    
    if 'vf_005' in vf_section:                          # If in Focal Vision
        
        fc_num = 0
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

        oc_connection_num = 0
        for ganglion in visual_field_dictionary[vf_section][3]:        # Define Focal Ganglion-Focal Occ. Relationship

            oc_connection_list = []
            synapse_values_list = []
            for _ in range(int(FOCAL_OCCIPITAL_PER_SECTION/9)):             # (REFORMAT) Focal Occ. / Num of Focal Ganglion = F. Occ. / F. Ganglion
                
                oc_connection_list.append(visual_field_dictionary[vf_section][4][oc_connection_num])
                oc_connection_num+=1

                synapse_values = 0.500
                synapse_values_list.append(synapse_values)

            synapse_relationships = [oc_connection_list, synapse_values_list]
            onn_map[ganglion] = synapse_relationships

        for occipital in visual_field_dictionary[vf_section][4]:        # Define Focal Occ.-Shape Specific Relationship

            ss_connection_list = []
            synapse_values_list = []
            for ssn in full_ssn_list:

                ss_connection_list.append(ssn)

                synapse_values = 0.500
                synapse_values_list.append(synapse_values)
            
            synapse_relationships = [ss_connection_list, synapse_values_list]
            onn_map[occipital] = synapse_relationships
                
    elif 'si' in vf_section:
        pass

    else:                                                             # Else in Peripheral Vision
        
        pr_num = 0
        one_of_three_px_count = 3
        for pixel in visual_field_dictionary[vf_section][0]:          # Define Pixel-Rod Relationship
            

            if pixel == 'px_009' or pixel == 'px_025' or pixel == 'px_057' or pixel == 'px_073':
                onn_map[pixel] = [[visual_field_dictionary[vf_section][1][pr_num], visual_field_dictionary[vf_section][1][-1]],
                                    [1, 1]]
                pr_num+=1
            
            elif pixel == 'px_016' or pixel == 'px_017' or pixel == 'px_018' or pixel == 'px_030' or pixel == 'px_033' or pixel == 'px_036' or pixel == 'px_046' or pixel == 'px_049' or pixel == 'px_052' or pixel == 'px_064' or pixel == 'px_065' or pixel == 'px_066':
                visual_field_dictionary[vf_section][1][-one_of_three_px_count]
                onn_map[pixel] = [[visual_field_dictionary[vf_section][1][pr_num], visual_field_dictionary[vf_section][1][-one_of_three_px_count]],
                                    [1, 1]]
                one_of_three_px_count-=1
                pr_num+=1
            else:
                onn_map[pixel] = [[visual_field_dictionary[vf_section][1][pr_num]],
                                    [1]]
                pr_num+=1

        copy_list_of_bipolar_cells = visual_field_dictionary[vf_section][2].copy()
        for photoreceptor in visual_field_dictionary[vf_section][1]:            # Define photoreceptor-Bipolar Relationship

            if 'pc' in photoreceptor:
                bp = copy_list_of_bipolar_cells.pop(1)
                onn_map[photoreceptor] = [[bp],
                                            [1]]
            else:
                onn_map[photoreceptor] = [copy_list_of_bipolar_cells,
                                            [1]]

        for bipolar in visual_field_dictionary[vf_section][2]:      # ONLY THE FIRST BP - Define Bipolar-Ganglion Relationship       
            
            # Connect each Bipolar Cell to Ganglion
            onn_map[bipolar] = [visual_field_dictionary[vf_section][3],
                                [1]]

        for ganglion in visual_field_dictionary[vf_section][3]:        # Define Ganglion-Occ. Relationship

            oc_connection_list = []
            synapse_values_list = []
            for occipital in visual_field_dictionary[vf_section][4]:    # Connect Ganglion to each Occipital Entry in Section
                
                oc_connection_list.append(occipital)

                synapse_values = 0.500
                synapse_values_list.append(synapse_values)

            synapse_relationships = [oc_connection_list, synapse_values_list]
            onn_map[ganglion] = synapse_relationships

        for occipital in visual_field_dictionary[vf_section][4]:        # Define Occ.-Shape Specific Relationship

            ss_connection_list = []
            synapse_values_list = []
            for ssn in full_ssn_list:

                ss_connection_list.append(ssn)

                synapse_values = 0.500
                synapse_values_list.append(synapse_values)
            
            synapse_relationships = [ss_connection_list, synapse_values_list]
            onn_map[occipital] = synapse_relationships


# bp_blacklist = ['bp_002', 'bp_004', 'bp_005', 'bp_006', 'bp_008', 'bp_010', 'bp_011', 'bp_012', 'bp_023', 'bp_024', 'bp_025', 
#     'bp_027', 'bp_029', 'bp_030', 'bp_031', 'bp_033']



for neuron in onn_map:
    if neuron in bp_blacklist:
        onn_map[neuron][1] = [0]

print(onn_map)