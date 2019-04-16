import cv2
import imutils
import numpy as np
import pkg_resources

import exercises.contour_shape_dedection.shapedetector as sd

image_filename = pkg_resources.resource_filename('resources', 'hand_drawn_bpmn_shapes.png')


def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, new_image = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    # show_image("flodfilled image", new_image)

    # Copy the thresholded image.
    im_floodfill = new_image.copy()


    h, w = new_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # show_image("flodfilled image", im_floodfill)

    # Combine the two images to get the foreground.
    new_image = new_image | im_floodfill_inv

    # show_image("foreground image", new_image)

    # blur the image slightly, and threshold it
    blurred = cv2.GaussianBlur(new_image, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    show_image("blurred image", thresh)

    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        shape = sd.detect(c)

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 0, 200), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)


def show_image(name, image):
  cv2.imshow(name, image)
  cv2.waitKey(0)


original_image = cv2.imread(image_filename, )
detect_shapes(original_image)
