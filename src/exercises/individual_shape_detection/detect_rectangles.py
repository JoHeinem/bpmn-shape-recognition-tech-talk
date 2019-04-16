import math

import cv2
import numpy as np
import pkg_resources

from exercises.individual_shape_detection.detect_corners import detect_corner_indexes

def calc_dist(p2, p1):
    return math.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)


def pool_close_corners(corner_idxs):
    result = np.array([[corner_idxs[0][0], corner_idxs[1][0]]])
    for ith_corner in range(len(corner_idxs[0])):
        p1 = np.array([corner_idxs[0][ith_corner], corner_idxs[1][ith_corner]])
        should_add = True
        # TODO:
        if gray[p1[0], p1[1]] == 255:
            should_add = False
        else:
            for p2 in result:
                if calc_dist(p2, p1) < 30:
                    should_add = False

        if should_add:
            result = np.vstack((result, p1))

    return result


def copy_array(array):
    array_copy = []
    for r in array:
        array_copy.append(r)
    array_copy = np.array(array_copy)
    return array_copy


def draw_rectangles():
    global idx_to_remove, can_continue, next_point, max_right, left_down, down, right_down, right, max_up, right_up, up, max_left, left_up, left, dist, dist_left, dist_bottom, dist_right, x, y, w, h, to_choose, rectangle_corners
    while np.size(rectangle_corners) != 0:
        upper_left_corner_idx = np.argmin(np.sum(rectangle_corners, axis=1))
        upper_left_corner_point = rectangle_corners[upper_left_corner_idx]

        # the corner point is not directly on the edge
        if gray[upper_left_corner_point[0] + 1, upper_left_corner_point[1]] != 0:
            upper_left_corner_point = [upper_left_corner_point[0], upper_left_corner_point[1] + 1]

        idx_to_remove = [upper_left_corner_idx]

        can_continue = True
        next_point = upper_left_corner_point
        max_right = 10
        while can_continue:
            left_down = gray[next_point[0] + 1, next_point[1] - 1]
            down = gray[next_point[0] + 1, next_point[1]]
            right_down = gray[next_point[0] + 1, next_point[1] + 1]
            right = gray[next_point[0], next_point[1] + 1]
            if down == 0:
                next_point = [next_point[0] + 1, next_point[1]]
            elif right_down == 0:
                next_point = [next_point[0] + 1, next_point[1] + 1]
            elif left_down == 0:
                next_point = [next_point[0] + 1, next_point[1] - 1]
            elif right == 0 and max_right > 0:
                max_right -= 1
                next_point = [next_point[0], next_point[1] + 1]
            else:
                can_continue = False

        # TODO: calc dist
        can_continue = True
        lower_left_corner = next_point
        max_up = 10
        while can_continue:
            right_up = gray[next_point[0] - 1, next_point[1] + 1]
            right = gray[next_point[0], next_point[1] + 1]
            right_down = gray[next_point[0] + 1, next_point[1] + 1]
            up = gray[next_point[0] - 1, next_point[1]]
            if right == 0:
                next_point = [next_point[0], next_point[1] + 1]
            elif right_up == 0:
                next_point = [next_point[0] - 1, next_point[1] + 1]
            elif right_down == 0:
                next_point = [next_point[0] + 1, next_point[1] + 1]
            elif up == 0 and max_up > 0:
                max_up -= 1
                next_point = [next_point[0] - 1, next_point[1]]
            else:
                can_continue = False
        # TODO: calc dist
        can_continue = True
        lower_right_corner = next_point
        max_left = 10
        while can_continue:
            left_up = gray[next_point[0] - 1, next_point[1] - 1]
            up = gray[next_point[0] - 1, next_point[1]]
            right_up = gray[next_point[0] - 1, next_point[1] + 1]
            left = gray[next_point[0], next_point[1] - 1]
            if up == 0:
                next_point = [next_point[0] - 1, next_point[1]]
            elif left_up == 0:
                next_point = [next_point[0] - 1, next_point[1] - 1]
            elif right_up == 0:
                next_point = [next_point[0] - 1, next_point[1] + 1]
            elif left == 0 and max_left > 0:
                max_left -= 1
                next_point = [next_point[0], next_point[1] - 1]
            else:
                can_continue = False

        can_continue = True
        upper_right_corner = next_point
        while can_continue:
            left_up = gray[next_point[0] - 1, next_point[1] - 1]
            left = gray[next_point[0], next_point[1] - 1]
            left_down = gray[next_point[0] + 1, next_point[1] - 1]
            if left == 0:
                next_point = [next_point[0], next_point[1] - 1]
            elif left_down == 0:
                next_point = [next_point[0] + 1, next_point[1] - 1]
            elif left_up == 0:
                next_point = [next_point[0] - 1, next_point[1] - 1]
            else:
                can_continue = False

        dist = calc_dist(upper_left_corner_point, next_point)
        dist_left = calc_dist(upper_left_corner_point, lower_left_corner)
        dist_bottom = calc_dist(lower_right_corner, lower_left_corner)
        dist_right = calc_dist(lower_right_corner, upper_right_corner)
        if dist < 40 < dist_bottom and dist_left > 40 and dist_right > 40:
            x1 = upper_left_corner_point[0]
            x2 = upper_right_corner[0]
            y1 = upper_left_corner_point[1]
            y2 = lower_left_corner[1]
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            w = int((upper_right_corner[1] + lower_right_corner)[1] / 2) - y
            h = int((lower_left_corner[0] + lower_right_corner[0]) / 2) - x
            cv2.rectangle(image, (y, x), (y + w, x + h), (0, 255, 0), 3)
        to_choose = [x for x in range(rectangle_corners.shape[0]) if x not in idx_to_remove]
        rectangle_corners = rectangle_corners[to_choose, :]


def draw_diamonds():
    global idx_to_remove, can_continue, next_point, max_left, left_down, down, left, max_up, right, right_down, right_up, max_right, up, left_up, dist, dist_left, dist_bottom, dist_right, x, y, w, h, to_choose, diamond_corners
    while np.size(diamond_corners) != 0:
        top_idx = np.argmin(diamond_corners, axis=0)[0]
        top_corner = diamond_corners[top_idx]

        # the corner point is not directly on the edge
        if gray[top_corner[0] + 1, top_corner[1]] != 0:
            top_corner = [top_corner[0] + 3, top_corner[1]]

        idx_to_remove = [top_idx]

        can_continue = True
        next_point = top_corner
        max_left = 20
        while can_continue:
            left_down = gray[next_point[0] + 1, next_point[1] - 1]
            down = gray[next_point[0] + 1, next_point[1]]
            left = gray[next_point[0], next_point[1] - 1]
            if left_down == 0:
                next_point = [next_point[0] + 1, next_point[1] - 1]
            elif down == 0:
                next_point = [next_point[0] + 1, next_point[1]]
            elif left == 0 and max_left > 0:
                max_left -= 1
                next_point = [next_point[0], next_point[1] - 1]
            else:
                can_continue = False

        can_continue = True
        left_corner = next_point
        max_down = 20
        max_up = 10
        while can_continue:
            down = gray[next_point[0] + 1, next_point[1]]
            right = gray[next_point[0], next_point[1] + 1]
            right_down = gray[next_point[0] + 1, next_point[1] + 1]
            right_up = gray[next_point[0] - 1, next_point[1] + 1]
            if right_down == 0:
                next_point = [next_point[0] + 1, next_point[1] + 1]
            elif right == 0:
                next_point = [next_point[0], next_point[1] + 1]
            elif down == 0 and max_down > 0:
                max_down -= 1
                next_point = [next_point[0] + 1, next_point[1]]
            elif right_up == 0 and max_up > 0:
                max_up -= 1
                next_point = [next_point[0] - 1, next_point[1] + 1]
            else:
                can_continue = False

        can_continue = True
        bottom_corner = next_point
        max_right = 20
        while can_continue:
            right = gray[next_point[0], next_point[1] + 1]
            up = gray[next_point[0] - 1, next_point[1]]
            right_up = gray[next_point[0] - 1, next_point[1] + 1]
            if right_up == 0:
                next_point = [next_point[0] - 1, next_point[1] + 1]
            elif up == 0:
                next_point = [next_point[0] - 1, next_point[1]]
            elif right == 0 and max_right > 0:
                max_right -= 1
                next_point = [next_point[0], next_point[1] + 1]
            else:
                can_continue = False

        can_continue = True
        right_corner = next_point
        max_up = 20
        max_down = 10
        while can_continue:
            left_up = gray[next_point[0] - 1, next_point[1] - 1]
            left = gray[next_point[0], next_point[1] - 1]
            up = gray[next_point[0] - 1, next_point[1]]
            left_down = gray[next_point[0] + 1, next_point[1] - 1]
            if left_up == 0:
                next_point = [next_point[0] - 1, next_point[1] - 1]
            elif left == 0:
                next_point = [next_point[0], next_point[1] - 1]
            elif up == 0 and max_up > 0:
                max_up -= 1
                next_point = [next_point[0] - 1, next_point[1]]
            elif left_down == 0 and max_down > 0:
                max_down -= 1
                next_point = [next_point[0] + 1, next_point[1] - 1]
            else:
                can_continue = False

        dist = calc_dist(top_corner, next_point)
        dist_left = calc_dist(top_corner, left_corner)
        dist_bottom = calc_dist(bottom_corner, left_corner)
        dist_right = calc_dist(bottom_corner, right_corner)
        if dist < 50 and dist_left > 40 and dist_bottom > 40 and dist_right > 40:
            x = top_corner[0]
            y = left_corner[1]
            w = right_corner[1] - y
            h = bottom_corner[0] - x
            cv2.rectangle(image, (y, x), (y + w, x + h), (0, 255, 0), 3)
        to_choose = [x for x in range(diamond_corners.shape[0]) if x not in idx_to_remove]
        diamond_corners = diamond_corners[to_choose, :]


image_filename = pkg_resources.resource_filename('resources', 'hand_drawn_bpmn_shapes.png')

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(image_filename)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corner_indexes = detect_corner_indexes(image)
rectangle_corners = pool_close_corners(corner_indexes)
diamond_corners = copy_array(rectangle_corners)

# draw_rectangles()
draw_diamonds()


for r in diamond_corners:
    image[r[0], r[1]] = [0, 0, 255]

cv2.imshow('dst', image)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
