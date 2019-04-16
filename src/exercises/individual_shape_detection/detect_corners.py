import cv2
import numpy as np
import pkg_resources

image_filename = pkg_resources.resource_filename('resources', 'hand_drawn_bpmn_shapes.png')

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(image_filename)


def detect_corner_indexes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # perform the corner harris detection
    gray = np.float32(gray)
    corners = cv2.cornerHarris(gray, 2, 5, 0.04)
    # result is dilated for marking the corners, not important
    corners = cv2.dilate(corners, None)
    # find the corner indexes
    return np.where(corners > 0.01 * corners.max())


if __name__ == "__main__":
    corner_indexes = detect_corner_indexes(image)

    # Threshold for an optimal value, it may vary depending on the image.
    image[corner_indexes] = [0, 0, 255]

    cv2.imshow('corners', image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
