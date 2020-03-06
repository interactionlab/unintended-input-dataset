import numpy as np
from constants import *
from take_processing_methods import *
from marker_processing_methods import *
import os
import multiprocessing


heatmap_size_x = 37 * 5 #185
heatmap_size_y = 52 * 5 #260
HEATMAP_SIZE_X = 37 * 5
HEATMAP_SIZE_Y = 52 * 5
HEATMAP_SIZE = np.array([HEATMAP_SIZE_X - 50, HEATMAP_SIZE_Y - 50])

def update_heatmap_front_back(hm_front, hm_back, frame_index, marker_id, phone_dimensions, df):
    """
    Creates finger movement heatmaps for every marker for a given task. 
    Returns: A list of marker heatmaps. Starts with Thumb and goes to Little Finger. For every finger, markers start with Fn and ends with MCP.
    """
    if not is_frame_available(frame_index, marker_id + "_X", df):
        return hm_front, hm_back

    frame = df.loc[frame_index]    
    marker = [frame[marker_id + "_X"], frame[marker_id + "_Y"], frame[marker_id + "_Z"]]  
    #marker = get_transformed_marker(frame_index, marker_id, df)
    
    if marker == None:
        return hm_front, hm_back
   
    # Flip cordinates to map on different 0/0 ???
    p = ((marker[0:2] - phone_dimensions[0:2]) * 1000) + HEATMAP_SIZE
    

    if p[0] < 0 or p[0] >= hm_front.shape[0] or np.isnan(p[0]):
        return hm_front, hm_back
    if p[1] < 0 or p[1] >= hm_front.shape[1] or np.isnan(p[1]):
        return hm_front, hm_back
    
    p = np.floor(p).astype(int)
    if(marker[2] > 0):
        #hm_front[p[0]][p[1]] = hm_front[p[0]][p[1]] + 1
        hm_front[p[0]][p[1]] += 1
    else:
        #hm_back[p[0]][p[1]] = hm_back[p[0]][p[1]] + 1
        hm_back[p[0]][p[1]] += 1
    
    return hm_front, hm_back

def update_heatmap_front_back_new(frame, marker_id, phone_dimensions):
    """
    Creates finger movement heatmaps for every marker for a given task. 
    Returns: A list of marker heatmaps. Starts with Thumb and goes to Little Finger. For every finger, markers start with Fn and ends with MCP.
    """
    if pd.isnull(frame[marker_id + "_X"]):
        return -1, -1, -1
        
    #frame = df.loc[frame_index]    
    marker = [frame[marker_id + "_X"], frame[marker_id + "_Y"], frame[marker_id + "_Z"]]
    if marker == None:
        return -1, -1 ,-1
    
    p = ((marker[:2] - phone_dimensions[:2]) * 1000) + HEATMAP_SIZE

    if p[0] < 0 or p[0] >= heatmap_size_x or np.isnan(p[0]):
        return -1, -1 ,-1
    if p[1] < 0 or p[1] >= heatmap_size_y or np.isnan(p[1]):
        return -1, -1 ,-1
    
    p = np.floor(p).astype(int)
    s = 1
    if(marker[2] > 0):
        return 0, p[0], p[1]
        #hm_front[p[0]][p[1]] = hm_front[p[0]][p[1]] + 1
        #hm_front[p[0]][p[1]] += 1
    else:
        #hm_back[p[0]][p[1]] = hm_back[p[0]][p[1]] + 1
        #hm_back[p[0]][] += 1
        return 1, p[0], p[1]
    

def create_heatmaps_one_finger (data):
    df, f, task, device_size = data
    heatmaps = []
    #brauche df informationen über task beginn - Frame 
    begin_scene = int(df[(df['Task']==task)]['Frame'].min())
    end_scene = int(df[(df['Task']==task)]['Frame'].max())
    for j in joints_of_finger(f):
        # print begin_scene, end_scene
        #hm_front = np.zeros((heatmap_size_x, heatmap_size_y))
        #hm_back = np.zeros((heatmap_size_x, heatmap_size_y))
        #for i in df.Frame[(df.Frame >= begin_scene) & (df.Frame <= end_scene)].tolist():
        #    hm_front, hm_back = update_heatmap_front_back(hm_front, hm_back, i, j, device_size, df)
        
        dfX = df[(df.Frame >= begin_scene) & (df.Frame <= end_scene)]
        ret  = dfX.apply(lambda x: update_heatmap_front_back_new(x, j, device_size), axis=1)
        hm_front = np.zeros((heatmap_size_x, heatmap_size_y))
        hm_back = np.zeros((heatmap_size_x, heatmap_size_y))

        for s, x, y in ret.tolist():
            if(s == 0):
                hm_front[x][y] += 1
            elif(s == 1):
                hm_back[x][y] += 1
                
        # f to int maybe?
        heatmaps.extend([(hm_front, hm_back, j, f)])
    return heatmaps

def create_heatmaps(task,df,device):
    """
    Creates finger movement heatmaps for every marker for a given task. 
    Returns: A list of marker heatmaps. Starts with Thumb and goes to Little Finger. For every finger, markers start with Fn and ends with MCP.
    """
    device_size = get_device_size(device)

    #x = df['PhoneX_Rotation']
    #y = df['PhoneY_Rotation']
    #z = df['PhoneZ_Rotation']
    #w = df['PhoneW_Rotation']
    #rot_matrices = np.array([
    #        [1.0-2*y*y-2*z*z, 2.0*x*y+2*w*z, 2.0*x*z - 2*w*y],
    #        [2.0*x*y - 2*w*z, 1.0-2*x*x-2*z*z, 2.0*y*z+2*w*x],
    #        [2.0*x*z+2*w*y, 2.0*y*z-2*w*x, 1.0-2*x*x-2*y*y]])#
    #rot_matrices = rot_matrices.T
    
    
    cpu_count = max(1, multiprocessing.cpu_count()-2)
    print("Using %i CPU cores" % cpu_count)
    pool = multiprocessing.Pool(cpu_count)
    lst = []
    for f in ALL_FINGER:
        lst.append([df, f, task, device_size])
    for df, f, task, device_size in lst:
        print("Finger %s, Task %s, device_size %s"%(f,task,device_size)) 
    heatmaps = pool.map(create_heatmaps_one_finger, lst)    
    pool.close()    
    return [item for sublist in heatmaps for item in sublist]
    
    #eatmaps = []
    #or f in ALL_FINGER:
    #   print(f)
    #   #brauche df informationen über task beginn - Frame 
    #   begin_scene = int(df[(df['Task']==task)]['Frame'].iloc[0])
    #   end_scene = int(df[(df['Task']==task)]['Frame'].iloc[-1])
    #   for j in joints_of_finger(f):
    #       # print begin_scene, end_scene
    #       hm_front = np.zeros((heatmap_size_x, heatmap_size_y))
    #       hm_back = np.zeros((heatmap_size_x, heatmap_size_y))
    #       heatmaps.extend([(hm_front, hm_back, j)])
    #                   
    #       for i in df.Frame[(df.Frame >= begin_scene) & (df.Frame <= end_scene)].tolist():
    #           hm_front, hm_back = update_heatmap_front_back(hm_front, hm_back, i, j, device_size, df, rot_matrices)
    return heatmaps

def load_heatmap_from_file(participant, device, cond, task):
    filename = "../heatmaps/" + cond + "/"+task+"_highres/P" + str(participant) + "_" + device + "_heatmaps.npy"
    if (os.path.isfile(filename)):
        return np.load(filename, encoding='latin1')
    else:
        return None


def get_heatmap(participant, device, task, marker_id):
    """
    Retrieve heatmap for a given participant, device, task (comfortable, stretched) and marker id. 
    Returns front_heatmap, back_heatmap
    """
    heatmaps = load_heatmap_from_file(participant, device, task)
    if heatmaps == None:
        return None, None
    else:
        return heatmaps[joints.index(marker_id)][0], heatmaps[joints.index(marker_id)][1]

def load_and_plot_heatmap(participant, device, task):
    heatmaps = load_heatmap_from_file(participant, device, task)

    # Plot heatmaps in 4x5 grid.
    fig, ax = plt.subplots(5, 4, figsize=(20, 30))
    for f in range(0, len(fingers)):
        for j in range(0, len(joints_of_finger(fingers[f]))):
            arr_index = f * 4 + j
            front_hm, back_hm, marker_name = heatmaps[arr_index]

            heatmap = back_hm
            if (f == 0):
                heatmap = front_hm

            ax[f][j].imshow(heatmap.T, interpolation='none', cmap='YlOrRd')

            device_width_px = get_device_size(device)[0] / 0.001
            device_height_px = get_device_size(device)[1] / 0.001
            rb_edge = (int(heatmap_size_x - 50), int(heatmap_size_y - 50))
            x = [rb_edge[0] - device_width_px, rb_edge[0], rb_edge[0], rb_edge[0] - device_width_px, rb_edge[0] - device_width_px]
            y = [rb_edge[1] - device_height_px, rb_edge[1] - device_height_px, rb_edge[1], rb_edge[1], rb_edge[1] - device_height_px]
            line, = ax[f][j].plot(x, y, 'b--')
            ax[f][j].set_title(marker_name + " (" + str(arr_index) + ")")
            ax[f][j].set_xlim([0,heatmap_size_x])
            ax[f][j].set_ylim([heatmap_size_y,0])

###############################################################################
###############################################################################
###############################################################################

def update_heatmap(hm, frame_index, marker_id, phone_dimensions, df):
    if not is_frame_available(frame_index, marker_id + "_X", df):
        return hm

    
    marker = get_transformed_marker(frame_index, marker_id, df)
    if marker == None:
        return hm
   
    p = marker[0:2]
    p = p / phone_dimensions[0:2]
    p = p * hm.shape

    if p[0] < 0 or p[0] >= hm.shape[0]:
        return hm
    if p[1] < 0 or p[1] >= hm.shape[1]:
        return hm
    
    p = np.floor(p).astype(int)
    hm[p[0]][p[1]] = hm[p[0]][p[1]] + 1
    
    return hm

def get_aggregated_heatmap(device, cond, task):
    """
    Create 20 heatmaps (every marker) aggregated over all participants for a device and task.
    E.g. comfortable zone of the Index Finger Fn for all participants while doing the "comfortable" task. 
    """
    added_heatmaps = []
    for i in range(0, 20):
        added_heatmaps.extend([[np.zeros((heatmap_size_x, heatmap_size_y)), np.zeros((heatmap_size_x, heatmap_size_y)), joints[i]]])

    directory = "../heatmaps/" + cond + "/" + task + "_highres/"
    for filename in os.listdir(directory):
        if device in filename:
            heatmaps = np.load(directory + filename, encoding='latin1')#
            for h in range(0, 20):
                added_heatmaps[h][0] = added_heatmaps[h][0] + (heatmaps[h][0]/heatmaps[h][0].sum())
                added_heatmaps[h][1] = added_heatmaps[h][1] + (heatmaps[h][1]/heatmaps[h][1].sum())
                
    return added_heatmaps

def get_aggregated_heatmap_excluded(device, task):
    """
    Create 20 heatmaps (every marker) aggregated over all participants for a device and task.
    E.g. comfortable zone of the Index Finger Fn for all participants while doing the "comfortable" task. 
    """
    added_heatmaps = []
    for i in range(0, 20):
        added_heatmaps.extend([[np.zeros((heatmap_size_x, heatmap_size_y)), np.zeros((heatmap_size_x, heatmap_size_y)), joints[i]]])

    directory = "../heatmaps/" + task + "_highres/"
    for filename in os.listdir(directory):
        if device in filename:
            participant = int(filename.split("_")[0][1:])
            device =filename.split("_")[1].split(".")[0]
            if (not (participant == 14)):
                heatmaps = np.load(directory + filename)
                for h in range(0, 20):
                    added_heatmaps[h][0] = added_heatmaps[h][0] + heatmaps[h][0]
                    added_heatmaps[h][1] = added_heatmaps[h][1] + heatmaps[h][1]
                
    return added_heatmaps