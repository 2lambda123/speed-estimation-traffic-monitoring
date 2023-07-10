import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, color, img_as_ubyte
from skimage.draw import ellipse_perimeter
from skimage.feature import canny
from skimage.transform import hough_ellipse


def draw_contours():
    image = cv2.imread("mask_test.png")

    # convert the image to grayscale format
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

    # cv2.imshow('Binary image', thresh)
    # cv2.waitKey(0)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(
        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
    )

    # draw contours on the original image
    image_copy = image.copy()
    # cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    index = 0

    list_len = [len(i) for i in contours]
    index = list_len.index(max(list_len))

    cv2.drawContours(
        image=image_copy,
        contours=contours[index],
        contourIdx=-1,
        color=(255, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    # see the results
    cv2.imshow("None approximation", image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_circles(image_path):
    # image = cv2.imread("extracted_cars/" + image_path)
    image = cv2.imread("mask_test.png")
    image_copy = image.copy()

    # convert the image to grayscale format
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    # thresh = cv2.bitwise_not(thresh)

    # circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=15, minRadius=0, maxRadius=0)

    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = False
    params.minArea = 30

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.4

    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.9

    # Set inertia filtering parameters
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(thresh)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(
        thresh,
        keypoints,
        blank,
        (255, 255, 0),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0,:]:
    #         cv2.circle(image_copy, (i[0], i[1]), i[2], (255, 255, 0), 2)

    cv2.imshow("None approximation", blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_circles_sklearn(image_path):
    # Load picture, convert to grayscale and detect edges
    # image_rgb = cv2.imread("extracted_cars/" + image_path)
    image_rgb = data.coffee()[0:220, 160:420]
    image_gray = color.rgb2gray(image_rgb)
    edges = canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)

    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(
        edges, accuracy=20, threshold=250, min_size=100, max_size=120
    )
    result.sort(order="accumulator")

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(
        ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True
    )

    ax1.set_title("Original picture")
    ax1.imshow(image_rgb)

    ax2.set_title("Edge (white) and result (red)")
    ax2.imshow(edges)

    plt.show()