import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.path import Path

def points_in_ellipse(ellipse, points):
    result_list = []
    for p in points:
        cos_angle = np.cos(np.radians(180.-ellipse[4]))
        sin_angle = np.sin(np.radians(180.-ellipse[4]))

        xc = p[0] - ellipse[0]
        yc = p[1] - ellipse[1]

        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle 

        rad_cc = (xct**2/(ellipse[2]/2.)**2) + (yct**2/(ellipse[3]/2.)**2)
        result_list.append(rad_cc)
            
    return result_list

def rasterize_ellipse(ellipse, raster):
    """
    Rasterizes ellipse into a given grid.
    PARAMS: 
        ellipse: (xc, yc, a, b, theta)
        grid: (width, height)
    """
    elli_mask = []
    for i in range(raster[0]):
        for j in range(raster[1]):
            elli_mask.append((i, j))

    elli_mask = points_in_ellipse(ellipse, elli_mask)
    return np.array([elli_mask]).reshape(raster[0], raster[1]) < 1.

def rasterize_contour(polygon_verts, raster):
    """
    Rasterizes contour into a given grid.
    PARAMS: 
        contour_vertices: List of [x,y] elements (can be retrieved by measure.find_contours[i])
        raster: (width, height)
    """
    poly_mask = []
    for i in range(raster[0]):
        for j in range(raster[1]):
            poly_mask.append((i, j))
            
    path = Path(polygon_verts)
    poly_mask = path.contains_points(poly_mask)
    return np.array([poly_mask]).reshape(raster[0], raster[1])

def polyToEllipseOverlap(ellipse_params, polygon_verts, heatmap_size):
    """
    Calculates overlap in cell units. 
    PARAMS: 
        ellipse: (xc, yc, width, height, theta_in_deg)
        polygon: List of [x,y] elements (can be retrieved by measure.find_contours[i])
        heatmap_size: (width, height)
    RETURNS:
        union: intersection area of ellipse and polygon
        f1:    F1 measure (accuracy without tn) when looking at overlapping at a classification problem
    """
    # Get rasterization of ellipse and polygon
    mask_ellipse = rasterize_ellipse(ellipse_params, heatmap_size)
    mask_polygon = rasterize_contour(polygon_verts, heatmap_size)

    # ellipse to polygon fitness calculation
    # accuracy = (tp + tn) / (tp + tn + fp + fn)
    union = np.logical_and(mask_ellipse, mask_polygon)
    tp = len(np.argwhere(union))
    fn = len(np.argwhere(mask_polygon)) - tp
    fp = len(np.argwhere(mask_ellipse)) - tp
    #print tp, fn, fp, len(np.argwhere(mask_ellipse)), len(np.argwhere(mask_polygon))
    if tp == 0:
        f1 = 0
    else: 
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn)
    
    return union, f1

def polyToPolyOverlap(e1, e2, heatmap_size):
    """
    Calculates overlap in cell units. 
    PARAMS: 
        ellipse1: (xc, yc, a, b, theta)
        ellipse2: (xc, yc, a, b, theta)
        heatmap_size: (width, height)
    RETURNS:
        union_mask: union of two polygons
        overlap: amount of overlapping cells
    """
    # Create ellipse mask
    mask_ellipse1 = rasterize_ellipse(e1, heatmap_size)
    mask_ellipse2 = rasterize_ellipse(e2, heatmap_size)
    
    union_mask = np.logical_and(mask_ellipse1, mask_ellipse2)

    return union_mask