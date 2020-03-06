# -*- coding: utf-8 -*-
import numpy as np
num_participants = 16
# Device List 
device_list = ['Samsung Galaxy S3 Mini', 'Samsung Galaxy S4', 'OnePlus One', 'Nexus 6']

def get_pixelsize_for_device(s):
    return {
        's3' : 0.1089,
        's4' : 0.0577,
        'opo': 0.0634, 
        'n6' : 0.0515
    }.get(s.lower(), None)

# Device names full
def get_full_device_name(s):
    return {
        's3' : device_list[0],
        's4' : device_list[1],
        'opo': device_list[2], 
        'n6' : device_list[3]
    }.get(s, None)


def get_filename_for_device(s):
    return {
        's3' : "S3",
        's4' : "S4",
        'opo': "OPO", 
        'n6' : "N6"
    }.get(s, None)

def get_device_diagonal(s):
    return {
        's3' : 10.16, #4 inches
        's4' : 12.7, #5 inches
        'opo': 13.9, #5.46 inches 
        'n6' : 15 #6 inches
    }.get(s, None)

def conv_phone_name(s):
    return {
        'GT-I8190' : 'S3',
        'GT-I9505' : 'S4',
        'A0001': 'OPO', 
        'Nexus 6' : 'N6'
    }.get(s, None)

def get_device_for_filename(s):
    return {
        'S3' : 's3',
        'S4' : 's4',
        'OPO': 'opo', 
        'N6' : 'n6'
    }.get(s, None)
def conv_name(s):
    print(s)
    return {
        'S3' : 's3',
        'S4' : 's4',
        'OPO': 'opo', 
        'N6' : 'n6'
    }.get(s, None)

# Device sizes
def get_device_size(device):
    return {
        's3' : np.array([0.063, 0.12155, 0.0099]), # Samsung S3 mini (121,55 mm × 63 mm × 9,9 mm)
        's4' : np.array([0.07, 0.137, 0.0079]), # Samsung Galaxy S4 (137 mm × 70 mm × 7,9 mm)
        'opo': np.array([0.0759, 0.1529, 0.0089]), # OnePlus One (152,9 mm × 75,9 mm × 8,9 mm)
        'n6' : np.array([0.083, 0.1593, 0.0101]) # Nexus 6 (159,3 mm × 83 mm × 10,1 mm)
    }.get(device, None)

def get_screen_coordinates(device):
    return {
        's3' : [[0, 5.23, 5.23, 0],[0,0,8.71,8.71]], # Samsung S3 mini (121,55 mm × 63 mm × 9,9 mm)
        's4' : [[0, 6.23, 6.23, 0],[0,0,11.07,11.07]], # Samsung Galaxy S4 (137 mm × 70 mm × 7,9 mm)
        'opo': [[0, 6.85, 6.85, 0],[0,0,12.18,12.18]], # OnePlus One (152,9 mm × 75,9 mm × 8,9 mm)
        'n6' : [[0, 7.42, 7.42, 0],[0,0,13.63,13.63]] # Nexus 6 (159,3 mm × 83 mm × 10,1 mm)
    }.get(device, None)

def get_screen_pos(device):

    return {

        's3' : np.array([(6.3-5.23)/2.0, (12.155-8.71)/2.0]), # Samsung S3 mini 8.71cm × 5.23

        's4' : np.array([(7.0-6.23)/2.0, (13.7/11.07)/2]), #  Samsung Galaxy S4 11.07cm × 6.23

        'opo': np.array([(7.59-6.85)/2.0, (15.29 - 12.18)/2.0]), # OnePlus One 12.18cm × 6.85

        'n6' : np.array([0.415, 1.15]) # Nexus 6 13.63cm x 7.42

    }.get(device, None)

# Fingers
fingers = ['Thumb', 'Index Finger', 'Middle Finger', 'Ring Finger', 'Little Finger']
ALL_FINGER = ['Thumb', 'Index Finger', 'Middle Finger', 'Ring Finger', 'Little Finger']

ALL_TASKS = ["WRITE", "READ", "Fitts"]

# Hand marker neighbors
joints = ['Thumb_Fn', 'Thumb_DIP', 'Thumb_PIP', 'Thumb_MCP',\
         'Index_Fn', 'Index_DIP', 'Index_PIP', 'Index_MCP',\
         'Middle_Fn', 'Middle_DIP', 'Middle_PIP', 'Middle_MCP',\
         'Ring_Fn', 'Ring_DIP', 'Ring_PIP', 'Ring_MCP',\
         'Little_Fn', 'Little_DIP', 'Little_PIP', 'Little_MCP',\
          'R_Shape_4','R_Shape_2','R_Shape_3','R_Shape_1',\
         'Wrist']

# For a given finger, return all joints
def joints_of_finger(x):
    return {
        'Thumb': ['Thumb_Fn', 'Thumb_DIP', 'Thumb_PIP', 'Thumb_MCP'],
        'Index Finger': ['Index_Fn', 'Index_DIP', 'Index_PIP', 'Index_MCP'],
        'Middle Finger': ['Middle_Fn', 'Middle_DIP', 'Middle_PIP', 'Middle_MCP'],
        'Ring Finger': ['Ring_Fn', 'Ring_DIP', 'Ring_PIP', 'Ring_MCP'],
        'Little Finger': ['Little_Fn', 'Little_DIP', 'Little_PIP', 'Little_MCP'],
        'Wrist': ['Wrist']
    }.get(x,None)

# return the joints' neighbors in the order from Fingernail, DIP, PIP and finally MCP
def neighborOf(x):
    return {
        'Thumb_Fn':'Thumb_DIP',
        'Thumb_DIP':'Thumb_PIP',
        'Thumb_PIP':'Thumb_MCP',
        'Index_Fn':'Index_DIP',
        'Index_DIP':'Index_PIP',
        'Index_PIP':'Index_MCP',
        'Middle_Fn':'Middle_DIP',
        'Middle_DIP':'Middle_PIP',
        'Middle_PIP':'Middle_MCP',
        'Ring_Fn':'Ring_DIP',
        'Ring_DIP':'Ring_PIP',
        'Ring_PIP':'Ring_MCP',
        'Little_Fn':'Little_DIP',
        'Little_DIP':'Little_PIP',
        'Little_PIP':'Little_MCP',
        'Wrist_1':'Wrist_2',
        'Wrist_3':'Wrist_4',
        'Wrist_5':'Wrist_6',
    }.get(x,None)