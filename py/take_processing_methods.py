from datetime import datetime
import time
import os
import pandas as pd
import numpy as np
from constants import * 

def read_take(take_file):
    """
    Reads the given take file and returns it in pandas format. 
    RETURN: Pandas dataframe, recording start as datetime
    """
    f_csv = open(take_file, 'r')
    marker_names = []   
    coordinates = [] 
    marker_type = []
    time_string = ""

    for i in range(0, 7):
        line = f_csv.readline()

        if i == 0:
            time_string = line.split(',')[9]
        if i == 3:
            marker_names = line.split(",")
        if i == 5:
            marker_type = line.split(",")
        if i == 6:
            coordinates = line.split(",")
    
    # Create the list of unique marker names to pass to the read_csv function.
    for i in range(0, len(marker_names)):
        newname = None
        for jname in joints:
            if jname.lower() in marker_names[i].lower() and not ("fka" in marker_names[i].lower()):
                newname = jname + "_" + str(coordinates[i])
                marker_names[i] = newname.strip()
        
        if newname == None:
            newname = marker_names[i] + "_" + str(coordinates[i])
            marker_names[i] = newname.strip()
            
        if marker_type[i].strip() != "Position":
            marker_names[i] = marker_names[i] + "_" + marker_type[i]
                
    marker_names[0] = 'Frame'
    marker_names[1] = 'Time'
    f_csv.close()
    
    # Importing one take (will be all takes later on)
    df = pd.read_csv(take_file, skiprows=list(range(0,7)), names=marker_names)
    df = df.set_index(df.Frame, drop=False)
    # Drop all irrelevant information
    
    
    # retrieve datetime and fix 12AM error of motive
    dt = datetime.strptime(time_string, '%Y-%m-%d %I.%M.%S.%f %p')
    if dt.hour == 0 and dt.strftime("%p") == "AM":
        dt = dt.replace(hour=12)
    
    return df, dt


def get_take_label(participant, device_name, take_start_datetime):
    """
    Get the recording labels of a given take (participant, device_name and take_start_datetime)
    RETURN: Recording labels as pandas data frame
    """
    take_labels = []
    
    # Times for take start
    ts_take = time.mktime(take_start_datetime.timetuple())

    # read in label file 
    ts = pd.read_csv('./timestamps/labels_task1_pt' + str(participant) + '.csv', names = ['status', 'timestamp', 'participant', 'device', 'finger', 'task'], index_col=False)

    
    c = 0
    for finger in ALL_FINGER:
        for task in ALL_TASKS:
            for status in ["Start", "Stop"]:
                # Condition start time (iloc[-1] to select the last entry)
                label_ts = ts[(ts['device'] == device_name) &
                    (ts['finger'] == finger) &
                    (ts['task'] == task) &
                    (ts['status'] == status)]['timestamp'].iloc[-1]
                
                label_dt = datetime.fromtimestamp(label_ts / 1000.0)
                frame_number = (((label_ts / 1000) - ts_take) * 240)
                
                t = []
                t.append(participant)
                t.append(finger)
                t.append(device_name)
                t.append(task)
                t.append(status)
                t.append(label_dt)
                t.append(frame_number)
                take_labels.append(t)


    # convert to dataframe
    s = np.array(take_labels)
    return pd.DataFrame(s, columns=['Participant', 'Finger', 'Phone', 'Task', 'Status', 'Take Time', 'Frame'])


def get_available_takes():
    """
    Returns a list of available takes in the form: [(participant1, phone1), (participant2, phone2), ...]
    """
    
    available_takes = []
    directory = "./recordings_clean/"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"): 
            participant = int(filename.split("_")[0][1:])
            device = get_device_for_filename(filename.split("_")[2].split(".")[0])
            available_takes.extend([(participant, device)])

    return available_takes

def check_marker_visibility(df, tl):
    """
    For the given take as pandas dataframe, and the recording labels as dataframe, this method prints
    the marker visibility rates.
    """
    avg = 0.0
    for f in ALL_FINGER:
        for j in joints_of_finger(f):
            begin_scene = int(tl[(tl['Finger'] == f) & (tl['Status'] == "Start") & (tl['Task'] == "Free exploration")]['Frame'].iloc[-1])
            end_scene = int(tl[(tl['Finger'] == f) & (tl['Status'] == "Stop") & (tl['Task'] == "Finger min. extended")]['Frame'].iloc[-1])

            marker = j + "_X"
            amount_nan = len(df[marker][begin_scene:end_scene][pd.isnull(df[marker])])
            amount_all = len(df[marker][begin_scene:end_scene])
            avg += 1.0 - float(amount_nan)/float(amount_all)
            # print j + ":", 1.0 - float(amount_nan)/float(amount_all), "(" + str(amount_nan) + " missing of " + str(amount_all) + ")"

    average = avg / 20
    # print "Average marker visibility:", average
    return average


def is_frame_available(frame_index, marker_id, df):
    """
    Checks if a frame is available (= rigid body and given marker is visible)
    RETURN: true if frame is available.
    """
    if pd.isnull(df[marker_id].ix[frame_index]):
        return False
    else:
        return True
    #q1 = pd.isnull(df[df.columns[2]][frame_index])
    #q2 = pd.isnull(df[df.columns[3]][frame_index])
    #q3 = pd.isnull(df[df.columns[4]][frame_index])
    #q4 = pd.isnull(df[df.columns[5]][frame_index])
    #marker = pd.isnull(df[marker_id][frame_index])

    #return not (q1 or q2 or q3 or q4 or marker)

def getReferenceFrame(df, tl, forFinger):
    begin_scene = int(tl[(tl['Finger'] == forFinger) & (tl['Status'] == "Start") & (tl['Task'] == "Free exploration")]['Frame'].iloc[-1])
    end_scene = int(tl[(tl['Finger'] == forFinger) & (tl['Status'] == "Stop") & (tl['Task'] == "Finger min. extended")]['Frame'].iloc[-1])

    # Get list of all relevant columns
    columns = ['Frame']
    for j in range(0, 20):
        columns.append(joints[j] + "_X")
        columns.append(joints[j] + "_Y")
        columns.append(joints[j] + "_Z")

    df1 = df[columns][begin_scene:end_scene]
    nan_list = df1.isnull().sum(axis=1).tolist()

    amount_nan = 0
    exit_loop = False
    nan_index = 0
    while (not exit_loop):
        try:
            nan_index_in_list = nan_list.index(amount_nan)
            nan_index = df1['Frame'][nan_index_in_list + begin_scene]
        except ValueError:
            amount_nan = amount_nan + 1
            continue

        exit_loop = True
        
    #for c in columns:
        #print df1[df1['Frame'] == nan_index][c].iloc[0]
        
    return df1[df1['Frame'] == nan_index]