import cv2 as cv
import numpy as np


roi = (200, 200, 300, 300) # use your ROI (x,y,w,h)

video_capture = cv.VideoCapture(0)

# aux buffers for GrabCut models
fgModel = np.zeros((1, 65), dtype="float")
bgModel = np.zeros((1, 65), dtype="float")

# video background
bckgd = None


print('Press SPACE to capture background.')
print('Make sure that no object is inside the image ROI (red box).')
print('You can reset the background at any time by pressing SPACE again.')

while True:
    _, frame = video_capture.read()
    img_roi = frame[roi[0] : roi[0] + roi[2], roi[1] : roi[1] + roi[3]]

    cv.rectangle(frame, roi, (0, 0, 255), 3)
    cv.imshow('Video', frame)

    # Hit 'ESC' on the keyboard to EXIT!
    key = cv.waitKey(5)
    if key == 27:
        break
    # Hit 'SPACE' to SET background
    elif key == 32:
        bckgd = cv.cvtColor(img_roi, cv.COLOR_BGR2GRAY)
    
    if bckgd is None:
        continue

    ##############################################################################
    ### GET OBJECT MASK IN CURRENT FRAME
    ##############################################################################

    # Get a first estimation of the foreground (in our case, the hand) mask
    mask = cv.absdiff(bckgd, cv.cvtColor(img_roi, cv.COLOR_BGR2GRAY))
    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    cv.imshow('Mask', mask)

    # Refine hand mask estimation with GrabCut
    if np.size(np.where(mask == 255)) == 0 or np.size(np.where(mask == 0)) == 0:
        continue
    mask_gc = np.zeros(img_roi.shape[:2], np.uint8)
    mask_gc[mask == 0] = 2
    mask_gc[mask == 255] = 3

    cv.grabCut(img_roi, mask_gc, None, bgModel, fgModel, 2, cv.GC_INIT_WITH_MASK)

    mask2 = np.where((mask_gc == 2)|(mask_gc == 0), 0, 1).astype('uint8')
    
    # fmask is the final hand mask!
    fmask = mask2 * 255
    cv.imshow('Mask2', fmask)

    ##############################################################################
    ### CHANGE BACKGROUND, KEEP FOREGROUND
    ##############################################################################

    new_bckgd = cv.imread('background.jpg')
    new_bckgd = cv.resize(new_bckgd, (img_roi.shape[1], img_roi.shape[0]))

    # Blur the mask to soften edges
    alpha = cv.GaussianBlur(fmask, (7, 7), 0)
    alpha = cv.cvtColor(alpha, cv.COLOR_GRAY2BGR)
    alpha = alpha.astype(float) / 255

    # Merge background and foreground object
    new_bckgd = new_bckgd.astype(float)
    new_bckgd = cv.multiply(1.0 - alpha, new_bckgd)
    new_bckgd = new_bckgd.astype(np.uint8)

    new_frgd = img_roi.astype(float)
    new_frgd = cv.multiply(alpha, new_frgd)
    new_frgd = new_frgd.astype(np.uint8)

    new_img = cv.add(new_bckgd, new_frgd)

    cv.imshow('final', new_img)




        