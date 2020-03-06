# -*- coding: utf-8 -*-
import os
from constants import *
from take_processing_methods import *
from marker_processing_methods import *
from heatmap_methods import *
from grasp_property_methods import *
import time

def conv_name(s):
    return {
        'SGS3' : 's3',
        'SGS4' : 's4',
        'OPO': 'opo', 
        'Nexus6' : 'n6'
    }.get(s, None)


toProcess = []

# get all takes in a folder
directory = "recordings\\"
comf_dir = "debug\\"
for filename in os.listdir(directory):
    if filename.endswith(".csv"): 
        participant = int(filename.split("_")[0][1:])
        device = conv_name(filename.split("_")[2].split(".")[0])
        hm_fname = "P" + str(participant) + "_" + str(device) + "_Index Finger.npy"
        if (not os.path.isfile(comf_dir + hm_fname)):
            toProcess.extend([(participant, device)])
    
print toProcess

avgMarkerVisibility = []
for participant, device in toProcess:
    ## take_processing_methods
    take_file = 'recordings/P' + str(participant) + '_Task0_' + get_filename_for_device(device) + '.csv'
    print "Importing " + take_file + "...",
    timer_start = time.clock()
    df, take_start_datetime = read_take(take_file)
    tl = get_take_label(participant, get_full_device_name(device), take_start_datetime)
    timer_end = time.clock()
    print " Done! " + "(" + str(timer_end - timer_start) + " sec.)"
    
    ## marker_processing_methods
    plot_and_save(df, tl, participant, device)
    print "Saved image."
    print "-----"

print "##############"