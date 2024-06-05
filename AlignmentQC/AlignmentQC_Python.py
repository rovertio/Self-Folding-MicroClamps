# Team 15: Micro Clamps 
# Implementation in python + CV + Camo

# Modules
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# Image accessing module
# --------------------------------------------
# Camera test
def get_available_cameras():
    available_cameras = []
    # check for 5 cameras
    for i in range(5):
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def imageCapture(available_cameras):
    cameras = available_cameras
    if cameras:
        print("Available cameras: ", cameras)
    else:
        print("No cameras found.")

    # Connect to video capture device
    cap = cv.VideoCapture(1)

    # Get a frame from the capture device.
    img_counter = 1
    for ii in range(2):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv.imshow("test", frame)
        
        k = cv.waitKey(0)
        if k%256 == 27:
            # Esc pressed
            print("Closing. . .")
            break
        elif k%256 == 32:
            # Space pressed
            img_name = "opencv_frame{}.png".format(img_counter)
            cv.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cap.release()
    cv.destroyAllWindows()

# --------------------------------------------


# Image processing module
# --------------------------------------------
# Canny feature detection + Image editing
def canny_detection(image):
    img = cv.imread(image, cv.IMREAD_COLOR)
    assert img is not None, "file could not be read, check with os.path.exists"

    crop_img=img[350:550, 670:800]
    crop_img=cv.rotate(crop_img,cv.ROTATE_90_COUNTERCLOCKWISE)

    edges = cv.Canny(crop_img,50,150)
    
    # cv.imshow("Edge Detection", edges)
    # cv.waitKey(0)
    # cv.destroyAllWindows

    return crop_img, edges


# Finding tip of the clamp jaw for reference
def tip_find(edges, clamp_number, res):    
    t_pt = 0
    prev_pt = 0

    # number of rows and columns
    scan_rows = np.size(edges,0)
    scan_col = np.size(edges,1)

    # threshold of points detected
    t_pt_th = clamp_number*2
    # fudge factor for tip detection
    # depends on the resolution for the image capture (pix/mm)
    tip_f = np.ceil(1.75*res)

    # Go down through rows of image
    for ii in range(scan_rows):
        if ii >= (0.8*scan_rows):
            print("Tip not found. Redo image capture.")
            tip = np.size(edges,1)
            break

        # adding points detected
        for jj in range(scan_col):
            if edges[ii,jj] == 255:
                t_pt = t_pt + 1
        
        if t_pt >= t_pt_th and prev_pt >= t_pt_th:
            tip = ii
            break
        else:
            prev_pt = t_pt
            t_pt = 0

    # Adding tip fudge factor        
    tip = np.int_(tip + tip_f)
    

    return tip

# --------------------------------------------



# Line computations
# --------------------------------------------
# Start and end point calculations
def line_pts(edges, tip):
    # start (1) -> base of clamp, end (2) -> tip of clamp jaw
    # y values for each point/edge deteced
    start_y = np.int_(np.ceil(0.8*np.size(edges,0)))
    end_y = np.int_(tip)

    # number of rows and columns
    # scan_rows = np.size(edges,0)
    scan_col = np.size(edges,1)

    jaw_x1 = np.zeros(scan_col)
    jaw_x2 = np.zeros(scan_col)
    prev_x1 = 1
    prev_x2 = 1
    point_tol = 2

    # Finding the start and end points
    for ii in range(scan_col):
        # (jaw_x1) Finding edge points at the base of the clamp
        if edges[start_y,ii] == 255 and abs(ii - prev_x1) >= point_tol:
            jaw_x1[ii] = ii
            prev_x1 = ii

        # (jaw_x2) Finding edge points at the tip of the clamp
        if edges[end_y,ii] == 255 and abs(ii - prev_x2) >= point_tol:
            jaw_x2[ii] = ii
            prev_x2 = ii

    # x values for each point/edge detected
    jaw_x1 = jaw_x1[jaw_x1 != 0]
    jaw_x2 = jaw_x2[jaw_x2 != 0]

    # If the vectors not the same size, lines cant be plot
    if np.size(jaw_x1) != np.size(jaw_x2):
        print("Error in image processing. Please retake shot")

    # Clamp x values at the base of the clamps
    clamp1_x1 = np.array(jaw_x1[(jaw_x1 - min(jaw_x1)) > 0.15*scan_col])
    clamp2_x1 = np.array(jaw_x1[(jaw_x1 - min(jaw_x1)) <= 0.15*scan_col])

    # Clamp x values at the tips of the clamps
    clamp1_x2 = np.array(jaw_x2[(jaw_x2 - min(jaw_x2)) > 0.15*scan_col])
    clamp2_x2 = np.array(jaw_x2[(jaw_x2 - min(jaw_x2)) <= 0.15*scan_col])

    # Clamp y values at the base of the clamps
    clamp1_y1 = np.array(start_y*np.ones(np.size(clamp1_x1)))
    clamp2_y1 = np.array(start_y*np.ones(np.size(clamp2_x1)))

    # Clamp y values at the tip of the clamps
    clamp1_y2 = np.array(end_y*np.ones(np.size(clamp1_x2)))
    clamp2_y2 = np.array(end_y*np.ones(np.size(clamp2_x2)))

    # edges closer to base
    c1_pts = np.array([clamp1_x1,
                     clamp1_x2,
                     clamp1_y1,
                     clamp1_y2])
    # edges closer to tips
    c2_pts = np.array([clamp2_x1,
                     clamp2_x2, 
                     clamp2_y1,
                     clamp2_y2])
    
    return start_y, end_y, c1_pts, c2_pts

# Angle calculations between lines 
def angle_cal(c1_pts, c2_pts):
    theta1 = np.zeros(2)
    theta2 = np.zeros(2)
    pt_num1 = np.size(c1_pts[0])
    pt_num2 = np.size(c2_pts[0])
    # print(pt_num1)
    # print(pt_num2)

    # Reference vector for use in clamp 1 calculations 
    v_ref1 = [c1_pts[1,0] - c1_pts[0,0], c1_pts[3,0] - c1_pts[2,0]]
    # Reference vector for use in clamp 2 calculations
    v_ref2 = [c2_pts[1,0] - c2_pts[0,0], c2_pts[3,0] - c2_pts[2,0]]

    # Computing angles through dot product
    for kk in range(pt_num1 - 1):
        # Obtianing vector for angle calculations   
        v_check1 = [c1_pts[1,kk+1] - c1_pts[0,kk+1], c1_pts[3,kk+1] - c1_pts[2,kk+1]]
        theta1[kk] = np.arccos(np.dot(v_ref1, v_check1)/(np.linalg.norm(v_ref1)*np.linalg.norm(v_check1)))
    
    for jj in range(pt_num2 - 1):
        # Obtianing vector for angle calculations   
        v_check2 = [c2_pts[1,jj+1] - c2_pts[0,jj+1], c2_pts[3,jj+1] - c2_pts[2,jj+1]]
        theta2[jj] = np.arccos(np.dot(v_ref2, v_check2)/(np.linalg.norm(v_ref2)*np.linalg.norm(v_check2)))

    theta1 = max(theta1[theta1 != 0])
    theta2 = max(theta2[theta2 != 0])

    return [theta1, theta2]

# --------------------------------------------


# Separation computations
# --------------------------------------------
def arc_len(ali_thresh, theta):
    # Constants for calculations
    jaw_len = 13        # legnth of clamp jaw (mm)
    jaw_width = 1.75    # width of jaw (mm)

    # Calculations for arc length
    tip_sep = np.zeros((2,np.size(theta)))
    for ii in range(np.size(theta)):
        sep = jaw_len*theta[ii]
        
        # Flag when exceed threshold
        if sep >= jaw_width*ali_thresh:
            tip_sep[1][ii] = 1
        else:
            tip_sep[1][ii] = 0
        
        tip_sep[0][ii]


if __name__ == '__main__':

    # Constants for analysis
    clamp_num = 2           # Number of clamps (2)
    res = 10                # Resolution (pix/mm)

    # Getting images from Camo and computer
    #imageCapture(get_available_cameras())
    image = "opencv_frame1.png"

    # Processing image to find edges
    crop_img, edges = canny_detection(image)

    # Finding points of edges
    tip = tip_find(edges, clamp_num, res)
    start_y, end_y, c1_pts, c2_pts = line_pts(edges, tip)
    # print(tip)
    # print("clamp one point matrix")
    # print(c1_pts)
    # print("clamp two point matrix")
    # print(c2_pts)

    # Finding angle measruements of the edges
    theta = angle_cal(c1_pts, c2_pts)


    # Separation esimated via arc length


    # Overlaying calculated edges with plot
    fig, ax = plt.subplots()
    crop_img = ax.imshow(crop_img)
    ax.set_title("Edges and angle values for clamps")

    # Plotting clamp one lines
    for ii in range(np.size(c1_pts[0])):
        ax.plot([c1_pts[0,ii], c1_pts[1,ii]], [c1_pts[2,ii], c1_pts[3,ii]], 
                ls='solid', linewidth=8, color = 'blue')

    # Plotting clamp two lines
    for jj in range(np.size(c2_pts[0])):
        ax.plot([c2_pts[0,jj], c2_pts[1,jj]], [c2_pts[2,jj], c2_pts[3,jj]], 
                ls='solid', linewidth=8, color = 'yellow')
        
    # Plotting offset lines for reference
    ax.hlines(start_y, max(c1_pts[1]), np.size(crop_img),  linewidths=3, 
              color='black', linestyle='dotted', label="Offset start")
    ax.hlines(end_y, max(c1_pts[1]), np.size(crop_img),  linewidths=3, 
              color='black', linestyle='dotted', label="Offset tip")
    
    # Annotations with angle values
    ax.text(50, 100, "Clamp force (gf): " + str(np.around(P_total, decimals=1)), color='g', fontsize=12)
    ax.text(50, 100, "Clamp force (gf): " + str(np.around(P_total, decimals=1)), color='g', fontsize=12)

    plt.show()

    # cv.imshow("Edge Detection", edges)
    # cv.waitKey(0) 
    # cv.destroyAllWindows() 