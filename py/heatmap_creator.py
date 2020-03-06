import os
from constants import *
from take_processing_methods import *
from marker_processing_methods import *
from heatmap_methods import *
import time
from pathlib import Path

def conv_name(s):
    return {
        'S3' : 's3',
        'S4' : 's4',
        'OPO': 'opo', 
        'N6' : 'n6'
    }.get(s, None)


def getAvgMarkerAvailability():
    toProcess = []

    # get all takes in a folder
    directory = "./recordings_clean/"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"): 
            participant = int(filename.split("_")[0][1:])
            device = conv_name(filename.split("_")[2].split(".")[0])
            toProcess.extend([(participant, device)])

    print(toProcess)

    avgMarkerVisibility = []
    for participant, device in toProcess:
        ## take_processing_methods
        take_file = './recordings_clean/P' + str(participant) + '_Task0_' + get_filename_for_device(device) + '.csv'
        print( "Importing " + take_file + "...",)
        timer_start = time.clock()
        df, take_start_datetime = read_take(take_file)
        tl = get_take_label(participant, get_full_device_name(device), take_start_datetime)
        timer_end = time.clock()
        print( " Done! " + "(" + str(timer_end - timer_start) + " sec.)")
        
        markerVis = check_marker_visibility(df, tl)
        print(markerVis)
        avgMarkerVisibility.append(markerVis)

    return avgMarkerVisibility


def generateHeatmaps(task):
    toProcess = []

    # get all takes in a folder
    #directory = "./recordings_todo/"
    directory = "./TransformedPickles/"
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"): 
            participant = int(filename.split("_")[0].replace("P",""))
            device = conv_name(filename.split("_")[1])
            cond = filename.split("_")[2].split(".")[0]
            toProcess.extend([(participant, cond, device)])

    #print(toProcess)

    #avgMarkerVisibility = []
    for participant, cond, device in toProcess:
        ## take_processing_methods
        take_file = './TransformedPickles/P' + str(participant)+"_" + get_filename_for_device(device)+"_"+cond + '.pkl'
        print("Importing /P" + str(participant)+"_" + get_filename_for_device(device)+"_"+cond + ".pkl")
        timer_start = time.clock()
        
        #df, take_start_datetime = read_take(take_file)
        df = pd.read_pickle(take_file)
        #tl = get_take_label(participant, get_full_device_name(device), take_start_datetime)
        timer_end = time.clock()
        print (" Done! " + "(" + str(timer_end - timer_start) + " sec.)")

        ## marker_processing_methods
        #markerVis = check_marker_visibility(df, tl)
        #print ("Average marker visibility: " + str(markerVis))
        #avgMarkerVisibility.extend([markerVis])
        #plot_frame(39579, df, device)
        #plot_frame_rb_coordinates(11060, df, device)

        ## heatmap_methods
        print("Creating heatmaps...")
        timer_start = time.clock()
        heatmaps = create_heatmaps(task=task, df=df, device=device)
        filenames = "P" + str(participant) + "_" + str(device) + "_heatmaps"
        timer_end = time.clock()
        print(" Done! " + "(" + str(timer_end - timer_start) + " sec.)")

        path = "./heatmaps/%s/%s_highres/" % (cond, task)
        Path(path).mkdir(parents=True, exist_ok=True)
        np.save("%s%s" % (path, filenames), heatmaps)
        print("Heatmaps saved to /heatmaps/%s/%s_highres/%s.npy"  % (cond, task, filenames))
  
        print ("-----")

    print ("##############")