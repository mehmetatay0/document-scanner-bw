import cv2 as cv
import numpy as np


def ScannerDocument(img):
    # image read here and resize process
    # Resize process be ease to us for finding contours
    # so percent value equal to 20
    img_orj = img.copy()
    percent = 20
    img = Resize(img, percent)

    # edge process
    edged = CannyEdge(img)

    cv.imwrite('process_images/edged.jpg', edged)

    # finding the document contour
    contour = ContourFinding(img, edged)

    contour_img = cv.drawContours(img.copy(), [contour], -1, (0, 255, 0), 5)
    cv.imwrite('process_images/contour_img.jpg', contour_img)

    # Perspective Transform on main image
    cropped_img = FourPointTransform(img_orj, contour, percent)

    cv.imwrite('process_images/cropped_img.jpg', cropped_img)

    return cropped_img


def Resize(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    resized = cv.resize(img, (width, height))
    return resized


def Convert2Points(pts, percent):
    convert = pts * (100 / percent)
    return convert


def CannyEdge(img):
    # grayscale and smoothing
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    # canny
    edged = cv.Canny(blur, 50, 100)
    return edged

def ContourFinding(img, edged):
    global contour
    _, contours, _ = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # the largest area of contours will be found
    # prev_area -> previous area is started zero
    prev_area = 0
    for c in contours:
        epsilon = 0.02 * cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, epsilon, True)
        area = cv.contourArea(c)
        if area > prev_area:
            contour = approx
            prev_area = area
        else:
            prev_area = area
    return contour


def SortThePoints(contour, percent):
    pts = contour.reshape(4, 2)
    sorted = np.zeros((4, 2), dtype="float32")

    # top - left and bottom - right points
    s = np.sum(pts, axis=-1)
    sorted[0] = pts[np.argmin(s)]
    sorted[2] = pts[np.argmax(s)]

    # top - right and bottom - left points
    diff = np.diff(pts, axis=-1)
    sorted[1] = pts[np.argmin(diff)]
    sorted[3] = pts[np.argmax(diff)]

    # that process convert img points to orj_img points
    sorted = Convert2Points(sorted, percent)
    return sorted


def FourPointTransform(img, contour, percent):
    # we need a sorted the points of contour
    sorted = SortThePoints(contour, percent)
    tleft, tright, bright, bleft = sorted

    # we compute the shape of new image
    # for Width
    widthA = np.sqrt(((bright[0] - bleft[0]) ** 2) + ((bright[1] - bleft[1]) ** 2))
    widthB = np.sqrt(((tright[0] - tleft[0]) ** 2) + ((tright[1] - tleft[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # for Height
    heightA = np.sqrt(((tright[0] - bright[0]) ** 2) + ((tright[1] - bright[1]) ** 2))
    heightB = np.sqrt(((tleft[0] - bleft[0]) ** 2) + ((tleft[1] - bleft[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Transform for Perspective
    PT = cv.getPerspectiveTransform(sorted, dst)

    # crop in main image
    cropped = cv.warpPerspective(img, PT, (maxWidth, maxHeight))
    return cropped


def ConvertBW(img):
    # convert to Black and White image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, threshed = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
    return threshed