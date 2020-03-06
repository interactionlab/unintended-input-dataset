from constants import *
from take_processing_methods import *
from marker_processing_methods import *
from heatmap_methods import *

def normalize(v):
    return v / np.linalg.norm(v)

def angleToDisplayPlane(direction):
    direction = normalize(direction)
    proj_dir = direction
    proj_dir[2] = 0
    proj_dir = normalize(proj_dir)
    return np.degrees(np.arccos(np.dot(direction, proj_dir)))
    
def angleToBottomEdge(direction):
    direction = normalize(direction)
    proj_dir = direction
    proj_dir[0] = 0
    proj_dir = normalize(proj_dir)
    return np.degrees(np.arccos(np.dot(direction, proj_dir)))

def isLittleFnSupporting(df, tl, device, forFinger):
    """
    Checks if the little finger is supporting the phone on the bottom corner.
    """
    threshold_x = -0.005
    threshold_y = 0.005
    
    frame_index = getReferenceFrame(df, tl, forFinger)['Frame'].iloc[0]
    trans_frame = get_transformed_markers(frame_index, df)
    device_dimensions = get_device_size(device)
    
    if not ('Little_Fn' in trans_frame and 'Little_DIP' in trans_frame and 'Little_PIP' in trans_frame and 'Little_MCP' in trans_frame):
        # not all markers are available for this test
        return "(missing)"
    finger = np.array([trans_frame['Little_Fn'], trans_frame['Little_DIP'], trans_frame['Little_PIP'], trans_frame['Little_MCP']])
    
    idx = -1
    for i in range(0,2):
        if finger[i][2] > 0 and finger[i+1][2] < 0:
            idx = i
#    print finger
#    print idx
    
    if idx == -1:
        return False
    
    direction = finger[idx] - finger[idx + 1]
    dist = -finger[idx + 1][2]
    direction = direction / direction[2] * dist
#    print finger[idx]
#    print finger[idx + 1]
#    print direction
    
    crossing = finger[idx + 1] + direction
#    print crossing
    
    if crossing[1] > (device_dimensions[1] - threshold_y) and crossing[0] > threshold_x:
        return True
    else:
        return False
    

## Winkel (Index_MCP-Little_MCP) zu Handy Plane 
def getMCPToPlaneAngle(df, tl, forFinger):
    """
    Calculates the angles between the Vector (Index_MCP - Little_MCP) to the bottom edge vector or the plane. 
    """
    frame_index = getReferenceFrame(df, tl, forFinger)['Frame'].iloc[0]
    trans_frame = get_transformed_markers(frame_index, df)

    
    if not ("Index_MCP" in trans_frame and "Little_MCP" in trans_frame):
        return "(missing)"
    
    index = np.array(trans_frame['Index_MCP'])
    little = np.array(trans_frame['Little_MCP'])
    
    direction = index - little
    
    return [angleToBottomEdge(direction), angleToDisplayPlane(direction)]
    # 1. retrieve reference frame ForFinger
    # 2. Create Index_MCP to Little_MCP Vector
    # 3. Calculate angle between (Index_MCP-Little_MCP) to bottom edge vector
    # Vermutung: Je "senkrechter" Index_MCP-Little_MCP runtergeht, desto mehr verschiebt sich die Comf Zone nach links


    
## Umfasst Finger die linke Kante?
def isFingerGraspingPhone(df, tl, graspingFinger, device, forFinger):
    threshold_x = -0.005
    threshold_y = 0.005
    
    device_dimensions = get_device_size(device)
        
    threshold_angle = 30
    frame_index = getReferenceFrame(df, tl, forFinger)['Frame'].iloc[0]
    
    trans_frame = get_transformed_markers(frame_index, df)
    
    if not (graspingFinger + "_Fn" in trans_frame and graspingFinger + "_DIP" in trans_frame):
        return "(missing)"
    
    fn = np.array(trans_frame[graspingFinger + '_Fn'])
    dip = np.array(trans_frame[graspingFinger + '_DIP'])
    
    if fn[1] > (device_dimensions[1] + threshold_y) or fn[1] < - threshold_y:
        return False
        
    if fn[0] >= threshold_x:
        return False
    #print fn[0]
    direction = normalize(fn - dip)
    proj_dir = direction
    proj_dir[2] = 0
    proj_dir = normalize(proj_dir)
    #print direction
    #print np.degrees(np.arccos(np.dot(direction, proj_dir)))
    if np.degrees(np.arccos(np.dot(direction, proj_dir))) > threshold_angle:
        return True
    else:
        return False
    # 1. retrieve reference frame forFinger
    # 2. Check if Finger_Fn is left of the left edge
    # 3. Calculate Angle between (Finger_Fn-Finger_DIP) to the left edge vector. If it is above a threshold, 
    #    then the finger is grasping

## Wie hoch ist der Index_MCP von der unteren Kante aus gesehen? 
def getIndexMCP_Y(df, tl, device, forFinger):
    """
    Returns the height of the Index_MCP from the bottom right corner.
    """
    frame_index = getReferenceFrame(df, tl, forFinger)['Frame'].iloc[0]
    trans_frame = get_transformed_markers(frame_index, df)
    device_dimensions = get_device_size(device)
    
    if not ("Index_MCP" in trans_frame):
        return "(missing)"
    
    return device_dimensions[1] - trans_frame['Index_MCP'][1]
    # Idea / TODO: Correlation between MCPHeight und Phone:Hand Ratio

## Wie hoch ist der Index_MCP von der unteren Kante aus gesehen? 
def getIndexMCP_X(df, tl, device, forFinger):
    """
    Returns the x position of the Index_MCP from the bottom right corner.
    """
    frame_index = getReferenceFrame(df, tl, forFinger)['Frame'].iloc[0]
    trans_frame = get_transformed_markers(frame_index, df)
    device_dimensions = get_device_size(device)

    if not ("Index_MCP" in trans_frame):
        return "(missing)"
    
    return device_dimensions[0] - trans_frame['Index_MCP'][0]
    # Idea / TODO: Correlation between MCPHeight und Phone:Hand Ratio

def getFingerCurvature(df, tl, finger, forFinger):
    frame_index = getReferenceFrame(df, tl, forFinger)['Frame'].iloc[0]
    
    trans_frame = get_transformed_markers(frame_index, df)
    
    
    if not (finger + '_Fn' in trans_frame and finger + '_PIP' in trans_frame and finger + '_MCP' in trans_frame):
        return "(missing)", "(missing)"
    
    fn = np.array(trans_frame[finger + '_Fn'])
    pip = np.array(trans_frame[finger + '_PIP'])
    mcp = np.array(trans_frame[finger + '_MCP'])
    
    left_dir  = fn - pip
    right_dir = mcp - pip
    
    angle = np.degrees(np.arccos(np.dot(normalize(left_dir), normalize(right_dir))))
    
    if angle > 170:
        return -1, angle
    
    up = np.cross(left_dir, right_dir)
    #print up
    #up_len = abs(np.linalg.norm(up))
    #if up_len < 0.000001:
    #    return -1
    
    
    up_dir = normalize(up)
    right_rect = normalize(np.cross(up_dir, right_dir))
    left_rect = normalize(np.cross(up_dir, left_dir))
    
    new_up = np.cross(left_rect, right_rect)
    
    
    
    right_mid = pip + (right_dir / 2)
    left_mid = pip + (left_dir / 2)
    
    t_left = np.linalg.det(np.array([(right_mid - left_mid), right_rect, new_up]) / np.square(np.linalg.norm(new_up)))
    
    
    center = left_mid + t_left * left_rect
    
    return np.linalg.norm(pip - center), angle
    
    # 1. retrieve reference frame forFinger
    # 2. Calculate angles between the bones
    # 3. Think of a metric TODO
    # Was ist der Abstand zwischen den PIP's und der Handy-Plane?
    
    
# Liefert den Abstand zwischen MCP zur linken oberen Ecke, sowie einen Punkt auf der Parabel zur linken oberen Ecke. 
def getGapBetweenMCPAndArea(df, tl, forFinger, Task):
    ref_frame = getReferenceFrame(df, tl, forFinger)
    # 1. Retrieve reference frame forFinger (to get MCP)
    # 2. Put the ellipse/function onto the backside of the phone
    # 3. Calculate distance between MCP to backside of the phone
    # 4. Return both. 
def plot_current_frame(df, device, forFinger):
    frame_index = getReferenceFrame(df, tl, forFinger)['Frame'].iloc[0]
    plot_frame_rb_coordinates(frame_index, df, device)




####################################################################################################
####################################################################################################
####################################################################################################
# BRESENHAM STUFF
####################################################################################################
####################################################################################################
####################################################################################################

def write(x, y, value, hm):
    if x < 0 or x >= len(hm) or y < 0 or y >= len(hm[0]):
        return
    hm[x][y] = value
    
def bresen_rd(x0, y0, x1, y1, value, hm):
    deltax = float(x1 - x0)
    deltay = float(y1 - y0)
    error = -1.0
    deltaerr = abs(deltay / deltax)#)    // Assume deltax != 0 (line is not vertical),
    y = y0

    for x in range(x0, x1 + 1):
        if x != x0 or y != y0:
            write(x, y, value, hm)
        error = error + deltaerr
        while error >= 0.0:
            if y <= y1:
                write(x, y, value, hm)
            y = y + 1
            error = error - 1.0
    return hm

def bresen_ur(x0, y0, x1, y1, value, hm):
    deltax = float(x1 - x0)
    deltay = float(y1 - y0)
    error = -1.0
    deltaerr = abs(deltax / deltay)#)    // Assume deltay != 0 (line is not vertical),
    x = x0

    for y in range(y0, y1 - 1, -1):
        if x != x0 or y != y0:
            write(x, y, value, hm)
        error = error + deltaerr
        while error >= 0.0:
            if x <= x1:
                write(x, y, value, hm)
            x = x + 1
            error = error - 1.0
    return hm

def bresen_lu(x0, y0, x1, y1, value, hm):
    deltax = float(x1 - x0)
    deltay = float(y1 - y0)
    error = -1.0
    deltaerr = abs(deltay / deltax)#)    // Assume deltax != 0 (line is not vertical),
    y = y0

    for x in range(x0, x1 - 1, -1):
        if x != x0 or y != y0:
            write(x, y, value, hm)
        error = error + deltaerr
        while error >= 0.0:
            if y >= y0:
                write(x, y, value, hm)
            y = y - 1
            error = error - 1.0
    return hm

def bresen_dl(x0, y0, x1, y1, value, hm):
    deltax = float(x1 - x0)
    deltay = float(y1 - y0)
    error = -1.0
    deltaerr = abs(deltax / deltay)#)    // Assume deltay != 0 (line is not vertical),
    x = x0

    for y in range(y0, y1 + 1, 1):
        if x != x0 or y != y0:
            write(x, y, value, hm)
        error = error + deltaerr
        while error >= 0.0:
            if x >= x1:
                write(x, y, value, hm)
            x = x - 1
            error = error - 1.0
    return hm

def bresen(x0, y0, x1, y1, value, hm):
    if x0 < x1 and y0 <= y1:
        return bresen_rd(x0, y0, x1, y1, value, hm)
    elif x0 <= x1 and y0 > y1:
        return bresen_ur(x0, y0, x1, y1, value, hm)
    elif x0 > x1 and y0 >= y1:
        return bresen_lu(x0, y0, x1, y1, value, hm)
    elif x0 >= x1 and y0 < y1:
        return bresen_dl(x0, y0, x1, y1, value, hm)
    elif x0 == x1 and y0 == y1:
        write(x0, y0, value, hm)
    else:
        return bresen(x1, y1, x0, y0, value, hm)

def clear_line_exclude_source(x, y, x1, y1, hm):
    bresen(x, y, x1, y1, 0, hm)
    hm[x][y] = 1
    
    
def outline(hm):
    origin_x = len(hm) - 10
    origin_y = len(hm[0]) - 10
    for x in range(len(hm)-1, 0, -1):
        for y in range(len(hm[0])-1, 0, -1):
            if hm[x][y] > 0:
                clear_line_exclude_source(x, y, origin_x, origin_y, hm)
                
                
def outline_from_center(hm, mcp):
    origin_x = int(mcp[1])
    origin_y = int(mcp[0])
    for x in range(len(hm)-1, 0, -1):
        for y in range(len(hm[0])-1, 0, -1):
            if hm[x][y] > 0:
                clear_line_exclude_source(x, y, origin_x, origin_y, hm)
                
def outline_reverse(hm, mcp):
    origin_x = mcp[1]
    origin_y = mcp[0]
    
    debug_points = []
    print("This is outline_reverse normal")
    for x in range(0, len(hm)-1, 1):
        for y in range(0, len(hm[0])-1, 1):
            if (hm[x][y] > 0):
                # Finding intersection with edges
                a_x = -origin_x / (x - origin_x)
                a_y = -origin_y / (y - origin_y)
                a = a_x
                
                if (abs(a_x) > abs(a_y)):
                    a = a_y

                # we know that they cannot be both negative
                if (a_x < 0):
                    a = a_y
                if (a_y < 0):
                    a = a_x
                    
                    
                target_x = int(origin_x + a * (x - origin_x))
                target_y = int(origin_y + a * (y - origin_y))
                
                debug_points.append([target_x, target_y])
                
                clear_line_exclude_source(x, y, target_x, target_y, hm)
                
    return np.array(debug_points)



def outline_reverse2(hm, mcp):
    mcp_x = mcp[1]
    mcp_y = mcp[0]
    
    debug_points = []
    
    for x in range(0, len(hm)-1, 1):
        for y in range(0, len(hm[0])-1, 1):
            if (hm[x][y] > 0):
      
                if (abs(x - mcp_x) < 0.0001):
                    clear_line_exclude_source(x, y, x, 0, hm)
                    continue
                
                m = (y - mcp_y) / (x - mcp_x)

                # for y = 0 (oberer corner)
                target_x = -(mcp_y / m) + mcp_x   # this makes y = 0
                target_y = 0 # m * (target_x - mcp_x) + mcp_y   # should be 0
                
                if (target_x < 0):
                    # for x = 0 (left corner)
                    target_x = 0
                    target_y = m * (target_x - mcp_x) + mcp_y
                    
                
                target_x = int(target_x)
                target_y = int(target_y)
                #print(target_x, target_y, mcp_x, mcp_y, x, y)
                
                debug_points.append([target_x, target_y])
                
                clear_line_exclude_source(x, y, target_x, target_y, hm)
                
    return np.array(debug_points)



def plot_and_save(df, tl, participant, device):
    fingers = ['Thumb', 'Index Finger', 'Middle Finger', 'Ring Finger', 'Little Finger'];
    for forFinger in fingers:
        frame_index = getReferenceFrame(df, tl, forFinger)['Frame'].iloc[0]
        supp = isLittleFnSupporting(df, tl, device, forFinger)
        isIndexGrasp = isFingerGraspingPhone(df, tl, 'Index', device, forFinger)
        isMiddleGrasp = isFingerGraspingPhone(df, tl, 'Middle', device, forFinger)
        isRingGrasp = isFingerGraspingPhone(df, tl, 'Ring', device, forFinger)
        isLittleGrasp = isFingerGraspingPhone(df, tl, 'Little', device, forFinger)
        
        mcpBot = getMCPToPlaneAngle(df, tl, forFinger)[0]
        mcpPlane = getMCPToPlaneAngle(df, tl, forFinger)[1]
        
        curvThumb = getFingerCurvature(df, tl, 'Thumb', forFinger)
        curvIndex = getFingerCurvature(df, tl, 'Index', forFinger)
        curvMiddle = getFingerCurvature(df, tl, 'Middle', forFinger)
        curvRing = getFingerCurvature(df, tl, 'Ring', forFinger)
        curvLittle = getFingerCurvature(df, tl, 'Little', forFinger)

        # Missing markers
        missing_markers = ""
        trans_frame = get_transformed_markers(frame_index, df)
        for f in fingers:
            for j in joints_of_finger(f):
                if not (j in trans_frame):
                    missing_markers = missing_markers + j + ", "
                    
        missing_markers = "(" + missing_markers + ")"
        
        grasp = ''.join(str(e) + "," for e in [isIndexGrasp, isMiddleGrasp, isRingGrasp, isLittleGrasp])
        curv = ''.join(str(e) + "," for e in [curvThumb, curvIndex, curvMiddle, curvRing, curvLittle])
        fig, ax = plot_frame_rb_coordinates(frame_index, df, device)
        ax.set_title("P" + str(participant) + "; " + device + "; " + forFinger + "; Missing: " + missing_markers + "\n supp = " + str(supp) + \
                     "; Grasp: " + grasp +\
                     "\nmcpToBot: " + str(mcpBot) + "; mcpToPlane: " + str(mcpPlane) + "; curvature: " + curv)
        fig.savefig('debug/p' + str(participant) + "_" + device + "_" + forFinger + ".png", dpi=fig.dpi)