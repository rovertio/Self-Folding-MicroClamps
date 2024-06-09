#!/usr/bin/env python
# Team 15: Micro Clamps 
# Implementation in python + CV + Camo

# Modules
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

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

def imageCapture(available_cameras, im_name):
    cameras = available_cameras
    if cameras:
        print("Available cameras: ", cameras)
    else:
        print("No cameras found.")

    # Connect to video capture device
    cap = cv.VideoCapture(1)

    # Get a frame from the capture device.
    img_name = []
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

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            img_nameii = "ForceTest_pic_no{}".format(im_name) + "_state{}.png".format(img_counter)
            img_name.append(img_nameii)
            cv.imwrite(img_nameii, frame)
            print("{} written!".format(img_nameii))
            img_counter += 1

    cap.release()
    cv.destroyAllWindows()

    return img_name

# ret, frame = cap.read()

# # Check if connected
# print(ret)

# # Write the frame from the video to a file
# cv.imwrite("test3.tiff",frame)

# Canny feature detection + Image editing
def canny_detection(image):
    img = cv.imread(image, cv.IMREAD_COLOR)
    assert img is not None, "file could not be read, check with os.path.exists"

    # crop_img=cv.rotate(sat_im,cv.ROTATE_90_CLOCKWISE)
    # crop_img=crop_img[0:350, 500:850]
    crop_img=img[100:230, 570:1000]

    sat_im = cv.cvtColor(crop_img, cv.COLOR_BGR2HSV)
    sat_im=sat_im[:,:,2]

    edges = cv.Canny(sat_im,35,150)
    #edges = cv.Canny(sat_im,20,170)
    
    # cv.imshow("Edge Detection", edges)
    # cv.waitKey(0)
    # cv.destroyAllWindows

    return crop_img, edges

# Finding the tips
# state is either Undeformed or Deformed
def tip_search(edges, tip_ap, state):
    match state:
        case "Undeformed":
            t_pt_th=4
            start_scan=np.size(edges,1)
        case "Deformed":
            t_pt_th=8
            start_scan=tip_ap+50
        case _:
            print('Choose Undeformed or Deformed as an argument.')
    
    t_pt = 0
    prev_pt = 0

    for ii in reversed(range(start_scan)):
        if ii <= (tip_ap - 200):
            print("Tip not found. Redo image capture. ")
            tip = np.size(edges,1)
            break
        for jj in range(np.size(edges,0)):
            if edges[jj,ii] == 255:
                t_pt = t_pt + 1
        
        if t_pt >= t_pt_th and prev_pt >= t_pt_th:
            tip = ii
            break
        else:
            prev_pt = t_pt
            t_pt = 0
    return tip

# Finding the start and end points of the beam
def y_diff(edges, start_off, tip):
    prev_y1 = 1
    prev_y2 = 1
    #tol_min = 0
    #tol_max = 200
    tol_min = 0
    tol_max = 100

    start_yf = np.zeros(np.size(edges,0))
    end_yf = np.zeros(np.size(edges,0))

    for ii in range(np.size(edges,0)):
        if edges[ii,start_off] == 255 and tol_min <= abs(ii - prev_y1) and abs(ii - prev_y1) <= tol_max:
            start_yf[ii] = ii
            prev_y1 = ii
        if edges[ii,tip] == 255 and tol_min <= abs(ii - prev_y2) and abs(ii - prev_y2) <= tol_max:
            end_yf[ii] = ii
            prev_y2=ii

    start_y=[0]*2
    end_y=[0]*2

    start_y[0] = np.min(start_yf[np.nonzero(start_yf)])
    start_y[1] = np.max(start_yf[np.nonzero(start_yf)])
    end_y[0] = np.min(end_yf[np.nonzero(end_yf)])
    end_y[1] = np.max(end_yf[np.nonzero(end_yf)])

    top_d = abs(end_y[0] - start_y[0])
    bot_d = abs(end_y[1] - start_y[1])
    return start_y, end_y, top_d, bot_d

def force_cal(ltest, top_d, dtop_d, bot_d, dbot_d, f_b):

    E_mod = 200*1000000000
    I_strip = (1/12)*(3.8*0.001)*(np.float_power(0.00064,3))
    L_strip = ltest*0.001
    N2gf = 101.972

    def_topp = abs(dtop_d - top_d)
    def_botp = abs(dbot_d - bot_d)

    def_top = def_topp*(1/res)*0.001
    def_bot = def_botp*(1/res)*0.001

    total_d = (def_top+def_bot)/2

    P_total = np.zeros(2)
    P_total[0] = N2gf*(total_d*3*E_mod*I_strip)/(np.float_power(L_strip,3))

    if P_total[0] < f_b[0]:
        P_total[1] = -1
    elif P_total[0] > f_b[1]:
        P_total[1] = 1
    else:
        P_total[1] = 0

    return P_total, total_d

def fig_plot(reg, deformed, tip, end_y, start_off, start_y, dend_y, dstart_y, P_total, res, f_b, im_name):

    # Gives the font styles for quality control
    flag_dict = {-1: ['red', 'normal'],
                 0: ['green', 'bold'],
                 1: ['red', 'bold']}
    
    fig, axis = plt.subplots(2,1)
    axis[0].imshow(reg)
    axis[0].set_title("Undeformed beam")
    axis[0].scatter([tip, tip], end_y, linewidths=1)
    axis[0].scatter([start_off,start_off], start_y, linewidths=1)
    axis[0].vlines(start_off, 0, np.size(reg,0),  linewidths=8, color='y', linestyle='dotted', label="Offset start")
    axis[0].vlines(tip, 0, np.size(reg,0), linewidths=8, color='b', linestyle='dotted', label="Jaw tips")
    axis[0].text(70, 35, "Threshold jaw force (gf): " + str(f_b), color='black', fontsize = 12)

    axis[1].imshow(deformed)
    axis[1].set_title("Deformed beam")
    axis[1].scatter([tip, tip], dend_y, linewidths=1)
    axis[1].scatter([start_off, start_off], dstart_y, linewidths=1)
    axis[1].vlines(start_off, 0, np.size(deformed,0), linewidths=8, color='y', linestyle='dotted', label="Offset start")
    axis[1].vlines(tip, 0, np.size(deformed,0), linewidths=8, color='b', linestyle='dotted', label="Jaw tips")

    axis[1].text(70, 35, "Clamp force (gf): " + str(np.around(P_total[0], decimals=1)), fontsize=12,
                 color=flag_dict[P_total[1]][0], fontweight=flag_dict[P_total[1]][1])
    
    #print(total_d)
    tip_sep = abs(dend_y[1] - dend_y[0])*(1/res)*0.001
    print(tip_sep)
    if np.around(tip_sep,decimals=5) < 0.0036:
        axis[1].text(70, 20, "Need to shorten spring (spring ends touch)", color='black', fontsize=12)


    plt.savefig("forcePlot_pic_no" + str(im_name)  + ".png")
    plt.show()

# --------------------------------------------



if __name__ == '__main__':
    # img = cv.imread("test2.tiff")
    # plt.imshow(img)
    # plt.show()
    # Constants for calculations
    res = np.ceil(310/55)                                                   # Approximate resolution
    ltest = np.float16(input("Measured value of spring length (mm): "))      # Spring length
    start_off = 40                                                           # Offset starting point
    tip_ap = res*ltest                                                      # Approximate spring length in pixels
    tip_off = 5                                                             # Offset used in tip calculations
    f_b = np.array([25, 45])

    try:
        while True:
            # Entering the identification number of the test/clamps
            test_num = input("Enter the test number for this iteration: ")

            # Getting images from Camo and computer
            img_name = imageCapture(get_available_cameras(), str(test_num))


            # --------------------------------------------
            # Image processing
            # reg = canny_detection("open_cv1.png")
            crop_reg, reg = canny_detection(img_name[0])
            crop_def, deformed = canny_detection(img_name[1])

            # Edge detection images for debugging
            cv.imshow("Edge Detection: undeformed", reg)
            cv.imshow("Edge Detection: deformed", deformed)
            cv.waitKey(0) 
            cv.destroyAllWindows()


            # --------------------------------------------
            # Undeformed value for beams
            # Picture of non-deflected beams
            # Getting differences in height between the top and bottom beams
            tip = tip_search(reg, tip_ap, "Undeformed") - tip_off
            start_y, end_y, top_d, bot_d = y_diff(reg, start_off, tip)


            # --------------------------------------------
            # Deformed value for beams
            # Picture of deflected beams
            # deformed = canny_detection("open_cv2.png")
            # Getting differences in height between the top and bottom beams
            dstart_y, dend_y, dtop_d, dbot_d = y_diff(deformed, start_off, tip)


            # --------------------------------------------
            # Force calculation for the clamp
            P_total, total_d = force_cal(ltest, top_d, dtop_d, bot_d, dbot_d, f_b)

            # Plotting points for verification
            fig_plot(crop_reg, crop_def, tip, end_y, start_off, start_y, dend_y, dstart_y, P_total, res, f_b, test_num)

    except KeyboardInterrupt:
        pass



