import cv2 as cv
import numpy as np
import math

shape_names = {None: 'circle',
               3: 'triangle',
               5: 'pentagon',
               6: 'hexagon'}


def endpoints_of_a_line(line):
    """
    This function takes a line object and calculates two endpoints and its angle
    :param line: Line object from cv.houghLines()
    :return: two endpoint and angle of line
    """
    rho = line[0][0]
    theta = line[0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
    return pt1, pt2, math.degrees(theta)


def center_point(p1, p2):
    """
    This function calculates middle point of 2 point
    :param p1: a point as (x, y)
    :param p2: a point as (x, y)
    :return: middle point as (x, y)
    """
    x1, y1 = p1
    x2, y2 = p2
    return (x1 + x2) // 2, (y1 + y2) // 2


def length_between_two_point(p1, p2):
    """
    This function calculates length between two point
    :param p1: a point as (x, y)
    :param p2: a point as (x, y)
    :return: length in pixel
    """
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def angle_between_two_line(mi, mj):
    """
    This function calculates angle between two lines
    :param mi: angle of line i in degree
    :param mj: angle of line j in degree
    :return: angle between two line in degree
    """

    def f(m):
        return 270 - m if m > 90 else 90 - m

    mi = f(mi)
    mj = f(mj)

    mi = math.tan(math.radians(mi))
    mj = math.tan(math.radians(mj))

    pay = (mi - mj)
    payda = (1 + mi * mj)
    if payda == 0:
        payda = 0.0000001
    return math.degrees(math.atan(pay / payda))


def eliminateSimilars(lines_, img):
    """
    This function finds similar lines and deletes all except one.
    :param lines_: output of cv.houghLines()
    :param img: original image to put output on it
    :return: None
    """
    if lines_ is None:  # There is no line
        return None

    # temp = img.copy() # For debugging
    lines = list(lines_)  # np.array is immutable. So, we convert to list to be able to delete items.

    i = 0
    j = 1
    while i < len(lines):
        while j < len(lines):
            pi1, pi2, mi = endpoints_of_a_line(lines[i])
            pj1, pj2, mj = endpoints_of_a_line(lines[j])

            # These comments for debugging
            # cv.line(temp, pi1, pi2, (0, 0, 255), 1, cv.LINE_AA)
            # cv.line(temp, pj1, pj2, (255, 0, 0), 1, cv.LINE_AA)
            # cv.imshow("test", temp)
            # cv.waitKey(1)
            # temp = img.copy()

            ci = center_point(pi1, pi2)
            cj = center_point(pj1, pj2)

            angle = angle_between_two_line(mi, mj)

            length = length_between_two_point(ci, cj)
            if abs(angle) < 10 and length < 40:
                # If the angle is similar and center of lines close each other, it means they are on same edge.
                # So, we don't need both. We should delete one of them.
                del lines[j]
            else:
                j += 1
        i += 1
        j = i + 1
    return lines


org = cv.imread("shapes6.jpeg")

gray = cv.cvtColor(org, cv.COLOR_BGR2GRAY)

blured = cv.GaussianBlur(gray, (7, 7), 0)
_, th = cv.threshold(blured, 180, 255, cv.THRESH_BINARY)

canny = cv.Canny(th, 50, 200, None, 3)

contours, _ = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

contours.sort(key=cv.contourArea, reverse=False)
contours.pop()  # To delete images itself

for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)

    cv.circle(org, (x + w // 2, y + h // 2), 3, (255, 0, 0), thickness=-1)  # Drawing center point

    piece = canny[y - 0:y + h + 0, x - 0:x + w + 0]
    # piece_show = org[y - 0:y + h + 0, x - 0:x + w + 0]

    lines = cv.HoughLines(piece, 1, np.pi / 180, 55)
    # lines = eliminateSimilars(lines, piece_show)

    if lines is None:  # If there is no edge
        cv.putText(org, 'circle', (x + w // 2, y + h // 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv.LINE_AA)

    elif len(lines) == 4:
        # If it has 4 edge it can be square or rectangle
        if abs(w / h - 1) < 0.2:  # For square with/height should be close to 1
            cv.putText(org, 'square', (x + w // 2, y + h // 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                       cv.LINE_AA)
        else:  # Otherwise it is rectangle
            cv.putText(org, 'rectangle', (x + w // 2, y + h // 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                       cv.LINE_AA)
    else:  # It can be named according to number of edges.
        cv.putText(org, shape_names[len(lines)], (x + w // 2, y + h // 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                   cv.LINE_AA)

cv.imshow("org", org)
cv.waitKey()
