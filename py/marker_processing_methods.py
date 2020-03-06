import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pylab

from constants import *
from take_processing_methods import *

def quaternion_matrix(quat):
    """
    Converts a quaternion to a rotation matrix.
    Source: http://fabiensanglard.net/doom3_documentation/37726-293748.pdf
    RETURN: Rotation matrix as numpy array.
    """
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]

    return np.array([
        [1.0-2*y*y-2*z*z, 2.0*x*y+2*w*z, 2.0*x*z - 2*w*y],
        [2.0*x*y - 2*w*z, 1.0-2*x*x-2*z*z, 2.0*y*z+2*w*x],
        [2.0*x*z+2*w*y, 2.0*y*z-2*w*x, 1.0-2*x*x-2*y*y]])

def plot_frame(frame_index, df, device):
    """
    Plots a given frame in world coordinates.
    """
    fig = plt.figure(figsize=(18,12))            
    ax = fig.add_subplot(111, projection='3d')

    # get frame and plot it. Fingers actual start at column 38 of the csv file
    marker_count = 0
    for f in xrange(26):
        marker_index = 38 + (f * 3)
        if not math.isnan(df[df.columns[marker_index]].ix[frame_index]):
            marker_count = marker_count + 1;
            ax.scatter(df[df.columns[marker_index]].ix[frame_index], df[df.columns[marker_index + 1]].ix[frame_index],\
                       df[df.columns[marker_index + 2]].ix[frame_index],edgecolor='b', s=50, c='#00FF00', marker='o', picker=5)

    ax = fig.gca(projection='3d')          
    plt.hold(True)

    # draw bones between joints
    for j in range(0, len(joints)):
        joint_name = joints[j]
        neighbor = neighborOf(joint_name)

        # check if the two joint marker are available
        if neighbor != None and not math.isnan(df[joint_name + "_X"].ix[frame_index]) and not math.isnan(df[neighbor + "_X"].ix[frame_index]):
            # draw a line between the two neighboring markers
            l_x = [df[joint_name + "_X"].ix[frame_index], df[neighbor + "_X"].ix[frame_index]]
            l_y = [df[joint_name + "_Y"].ix[frame_index], df[neighbor + "_Y"].ix[frame_index]]
            l_z = [df[joint_name + "_Z"].ix[frame_index], df[neighbor + "_Z"].ix[frame_index]]
            ax.plot(l_x, l_y, l_z, c='#00FF00')

    # Get rotation matrix from the rigid body quaternion
    q1 = df[df.columns[2]].ix[frame_index]
    q2 = df[df.columns[3]].ix[frame_index]
    q3 = df[df.columns[4]].ix[frame_index]
    q4 = df[df.columns[5]].ix[frame_index]
    rot_matrix = quaternion_matrix(np.array([q1, q2, q3, q4]))

    # Calculate and plot the phone (plane) based on given rigid body and device size. 
    device_size = get_device_size(device)
    dir_x = np.multiply([rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2]], device_size[0])
    dir_y = np.multiply([rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2]], 0.4)
    dir_z = np.multiply([rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2]], device_size[1])

    # Retrieve pivot point of the rigid body 
    rb_pivot =[df[df.columns[6]].ix[frame_index], df[df.columns[7]].ix[frame_index], df[df.columns[8]].ix[frame_index]]
    
    # Draw rigid body as a plane (representing the smartphone)
    x = [rb_pivot[0], dir_x[0] + rb_pivot[0], dir_x[0] + dir_z[0] + rb_pivot[0], dir_z[0] + rb_pivot[0]] 
    y = [rb_pivot[1], dir_x[1] + rb_pivot[1], dir_x[1] + dir_z[1] + rb_pivot[1], dir_z[1] + rb_pivot[1]] 
    z = [rb_pivot[2], dir_x[2] + rb_pivot[2], dir_x[2] + dir_z[2] + rb_pivot[2], dir_z[2] + rb_pivot[2]] 
    verts = [zip(x, y, z)]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.5))
    plt.hold(True)
    ax.scatter(x, y, z, edgecolor='b',c='#FF0000', marker='o', picker=5)

    # Display settings 
    ax.set_xlim3d((-0.5,0.5))
    ax.set_ylim3d((-0.5,0.5))
    ax.set_zlim3d((0,0.5))
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.view_init(azim=79, elev=0) 
    #ax.view_init(azim=169, elev=34) # shows which markers are in front and in the back of the phone  

    def onpick(event):
        ind = event.ind[0]
        x, y, z = event.artist._offsets3d
        print(x[ind], y[ind], z[ind])

    cid = fig.canvas.mpl_connect('pick_event', onpick)


def inv_trans(p, pos, rot):
    """
    Transform a marker into the rigid body coordinate system
    """
    p = np.subtract(p, pos)
    p = np.dot(p, rot.T)
    p_changed_dims = [p[0], p[2], p[1]]
    return p_changed_dims


# get list of markers transformed into the rigid body coordinate system
def get_transformed_marker(frame_index, marker_id, df):#, rot_matrices):
    #q1 = df[df.columns[2]].ix[frame_index]
    #q2 = df[df.columns[3]].ix[frame_index]
    #q3 = df[df.columns[4]].ix[frame_index]
    #q4 = df[df.columns[5]].ix[frame_index]
    #
    #if (q1 == None or q2 == None or q3 == None or q4 == None):
    #    return None
    #
    #rot_matrix = quaternion_matrix(np.array([q1, q2, q3, q4]))

    #rot_matrix = rot_matrices[df.index.get_loc(frame_index)].T

    # Pivot point of rigid body in global coordinates
    
    frame = df.loc[frame_index]    
    rb_pivot =[frame["PhoneX"], frame["PhoneY"], frame["PhoneZ"]]
    marker = [frame[marker_id + "_X"], frame[marker_id + "_Y"], frame[marker_id + "_Z"]]  
    rot_matrix = frame.RotationMatrix
    marker = inv_trans(marker, rb_pivot, rot_matrix)

    return marker


def plot_frame_rb_coordinates(frame_index, df, device):
    """
    Plot a frame of transformed markers (into the rigid body coordinate system).
    """
    # rotation matrix
    q1 = df[df.columns[2]].loc[frame_index]
    q2 = df[df.columns[3]].loc[frame_index]
    q3 = df[df.columns[4]].loc[frame_index]
    q4 = df[df.columns[5]].loc[frame_index]
    rot_matrix = quaternion_matrix(np.array([q1, q2, q3, q4]))
    
    # Pivot point of rigid body in global coordinates
    rb_pivot =[df[df.columns[6]].loc[frame_index], df[df.columns[7]].loc[frame_index], df[df.columns[8]].loc[frame_index]]
    
    transformed_markers = get_transformed_markers(frame_index, df)

    fig = plt.figure(figsize=(18,12))            
    ax = fig.add_subplot(111, projection='3d')
            
    # get frame and plot it. Fingers actually start at column 38 of the csv file
    for j in joints:
        if is_frame_available(frame_index, j + "_X", df):
            marker = get_transformed_marker(frame_index, j, df)

            ax.scatter(marker[0], marker[1], marker[2], edgecolor='b',c='#FFFFFF', marker='o')

    # draw bones between joints
    for j in range(0, len(joints)):
        joint_name = joints[j]
        neighbor = neighborOf(joint_name)

        # check if the two joint marker are available
        if (joint_name in transformed_markers and neighbor in transformed_markers):
            joint = get_transformed_marker(frame_index, joint_name, df)
            neighbor = get_transformed_marker(frame_index, neighbor, df)
            
            l_x = [joint[0], neighbor[0]]
            l_y = [joint[1], neighbor[1]]
            l_z = [joint[2], neighbor[2]]

            ax.plot(l_x, l_y, l_z)

    # Calculate and plot the phone (plane) based on given rigid body and device size. 
    device_size = get_device_size(device)
    dir_x = np.multiply([rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2]], device_size[0])
    dir_y = np.multiply([rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2]], 0.4)
    dir_z = np.multiply([rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2]], device_size[1])

            
    # Draw rigid body as a plane (representing the smartphone)
    x = [rb_pivot[0], dir_x[0] + rb_pivot[0], dir_x[0] + dir_z[0] + rb_pivot[0], dir_z[0] + rb_pivot[0]] 
    y = [rb_pivot[1], dir_x[1] + rb_pivot[1], dir_x[1] + dir_z[1] + rb_pivot[1], dir_z[1] + rb_pivot[1]] 
    z = [rb_pivot[2], dir_x[2] + rb_pivot[2], dir_x[2] + dir_z[2] + rb_pivot[2], dir_z[2] + rb_pivot[2]] 
    
    
    verts = [zip(x, y, z)]
    for j in range(0, len(verts[0])):
        verts[0][j] = inv_trans(verts[0][j], rb_pivot, rot_matrix)

    ax.add_collection3d(Poly3DCollection(verts, alpha=0.5))
    plt.hold(True)
    #ax.scatter(x, y, z, edgecolor='b',c='#FF0000', marker='o', picker=5)
    
    ax.set_xlim3d((-0.5,0.5))
    ax.set_ylim3d((-0.5,0.5))
    ax.set_zlim3d((0.5,-0.5))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=-90, elev=-90) 
    
    return fig, ax


###############################################################################
###############################################################################
###############################################################################

def get_transformed_markers(frame_index, df):
    """
    Get list of markers transformed into the rigid body coordinate system
    """
    q1 = df[df.columns[2]].ix[frame_index]
    q2 = df[df.columns[3]].ix[frame_index]
    q3 = df[df.columns[4]].ix[frame_index]
    q4 = df[df.columns[5]].ix[frame_index]
    
    if (q1 == None or q2 == None or q3 == None or q4 == None):
        return
    
    rot_matrix = quaternion_matrix(np.array([q1, q2, q3, q4]))

    # Pivot point of rigid body in global coordinates
    rb_pivot =[df[df.columns[6]].ix[frame_index], df[df.columns[7]].ix[frame_index], df[df.columns[8]].ix[frame_index]]

    
    transformed_markers = {}
    for f in xrange(26):
        marker_index = 38 + (f * 3)
        if not math.isnan(df[df.columns[marker_index]].ix[frame_index]):
            marker = [df[df.columns[marker_index]].ix[frame_index], df[df.columns[marker_index + 1]].ix[frame_index], df[df.columns[marker_index + 2]].ix[frame_index]]
            marker = inv_trans(marker, rb_pivot, rot_matrix)
            
            transformed_markers[df.columns[marker_index].replace("_X", "")] = marker
    return transformed_markers