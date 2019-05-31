import cv2
import numpy as np
import copy as cp
import time
from geometry import *

# The camera identifier as a video capturing device
CAMERA_ID = 1
# The resolution of map after perspective transformation
MAP_SIZE = (640, 480)
# The center of starting-point searching circle
ROUTE_BEGIN_POSITION = (0, 0)

# Minimum size for contour detection
MIN_CONTOUR_LENGTH = 20
# Threshold for route lines (color in lines should be lower than these thresholds)
ROUTE_SAT_RANGE = [0, 50]
ROUTE_VALUE_RANGE = [0, 120]
# The kernel size of morphology effect to routes
CLOSE_KERNEL_SIZE = 25
EROSION_KERNEL_SIZE = 15
# Channel threshold for marks
RED_HUE_RANGE = [0, 20]
GREEN_HUE_RANGE = [40, 80]
BLUE_HUE_RANGE = [90, 130]
# Saturation threshold of car mark
MARK_SAT_RANGE = [100, 255]
MARK_VALUE_RANGE = [100, 255]
# Search range of new keypoints
SEARCH_RANGE = 300
# The minimum distance of a newly added route point to existing points
CLOSEST_MIN_DIST = 50

# Use test image or video capture
TEST_WITH_IMAGE = True
# Decide whether to show certain images
SHOW_CORNER_MASK = False
SHOW_TRANSFORMED = False
SHOW_DETECTED_LINES = False
SHOW_MERGED_LINES = False
SHOW_KEYPOINTS = True
SHOW_HEAD_MARK = False
SHOW_TAIL_MARK = False
SHOW_CAR_POS = True

ASCII_ENTER = 13
ASCII_ESCAPE = 27

class RouteMap:
    def __init__(self):
        if not TEST_WITH_IMAGE:
            self.video = cv2.VideoCapture(CAMERA_ID)
        self.pixelBnd = np.float32([[0, 0], [MAP_SIZE[0], 0], [0, MAP_SIZE[1]], [MAP_SIZE[0], MAP_SIZE[1]]])
        self.firstRun = True

    def capture(self, updateMat):
        # Monitor tweak of camera and board position
        if self.firstRun:
            self.firstRun = False
            print("Please tweak the placement of camera and board")
            while True:
                self._read()
                self._findCorners()
                if cv2.waitKey(100) == ASCII_ENTER:
                    cv2.destroyAllWindows()
                    break

        # Read image and update matrix if required
        self._read()
        if updateMat:
            ok = False
            while not ok:
                fourCorners, ok = self._findCorners()
            self.matrix = cv2.getPerspectiveTransform(fourCorners, self.pixelBnd)

        # Apply perspective transformation to original image
        self.map = cv2.warpPerspective(self.original, self.matrix, MAP_SIZE)

        # Show tranformed image
        if SHOW_TRANSFORMED:
            cv2.imshow("Transformed Map", self.map)

        return True

    def _findCorners(self):
        ## Create corner contour mask
        blurred = cv2.GaussianBlur(self.original, (5, 5), 0)
        cornMask = createMask(blurred, BLUE_HUE_RANGE, MARK_SAT_RANGE, MARK_VALUE_RANGE)
        if SHOW_CORNER_MASK:
            cv2.imshow("Corner Mask", cornMask)

        # Find contours of corner objects
        contours, _ = cv2.findContours(cornMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Filter out contours with too short perimeter
        contours = filter(lambda cnt: len(cnt) >= MIN_CONTOUR_LENGTH, contours)

        # Extract center point of corner mark
        corners = [findContourCenter(cnt) for cnt in contours]

        # Visualize corners of the road map
        self.cornerPlot = cp.copy(self.original)
        for pos in corners:
            cv2.circle(self.cornerPlot, pos, 5, (0, 0, 255), thickness=-1)
        cv2.imshow("Corners", self.cornerPlot)

        # Raise error if corners are too few
        if len(corners) < 4:
            print("Can't detect corners of route map.")
            return None, False # try again next time

        # Get perspective matrix
        fourCorners = sorted(corners[:4], key=lambda p: p[0]+p[1])
        if fourCorners[1][1] > fourCorners[2][1]:
            fourCorners[1], fourCorners[2] = fourCorners[2], fourCorners[1]
        return np.float32(fourCorners), True

    def _read(self):
        if TEST_WITH_IMAGE:
            self.original = cv2.imread("img/rotate.jpg")
        else:
            _, self.original = self.video.read()

    def findRoute(self):
        # Create track bar for adjusting parameters of morphology transformation
        cv2.namedWindow("Route Morphology")
        cv2.namedWindow("Key Positions")
        keyPts = self._findRoute()

        def onChangeSat(x):
            ROUTE_SAT_RANGE[1] = x
            keyPts = self._findRoute()
        cv2.createTrackbar("Saturation", "Route Morphology", ROUTE_SAT_RANGE[1], 255, onChangeSat)
        def onChangeValue(x):
            ROUTE_VALUE_RANGE[1] = x
            keyPts = self._findRoute()
        cv2.createTrackbar("Value", "Route Morphology", ROUTE_VALUE_RANGE[1], 255, onChangeValue)
        def onChangeClose(x):
            global CLOSE_KERNEL_SIZE
            CLOSE_KERNEL_SIZE = x
            keyPts = self._findRoute()
        cv2.createTrackbar("Close", "Route Morphology", CLOSE_KERNEL_SIZE, 50, onChangeClose)
        def onChangeErosion(x):
            global EROSION_KERNEL_SIZE
            EROSION_KERNEL_SIZE = x
            keyPts = self._findRoute()
        cv2.createTrackbar("Erosion", "Route Morphology", EROSION_KERNEL_SIZE, 50, onChangeErosion)

        while True:
            if cv2.waitKey(25) == ASCII_ENTER:
                break
        cv2.destroyAllWindows()

        return keyPts

    def _findRoute(self):
        # Create road mask
        blurred = cv2.GaussianBlur(self.map, (3, 3), 0)
        roads = createMask(blurred, (0, 255), ROUTE_SAT_RANGE, ROUTE_VALUE_RANGE)
        roads = cv2.morphologyEx(roads, cv2.MORPH_CLOSE, np.ones((CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)))
        # Erode image (make roads appear narrower)
        roads = cv2.erode(roads, np.ones((EROSION_KERNEL_SIZE, EROSION_KERNEL_SIZE)), 
            iterations=1)
        cv2.imshow("Route Morphology", roads)

        # Hough line transform
        endpts = cv2.HoughLinesP(roads, 1, np.pi/180, 30, minLineLength=100, maxLineGap=10)
        if endpts is None:
            return None
        # Convert to lines representation
        lines = list(map(lambda l: Line(tuple(l[0])), endpts))

        # Visualize lines
        if SHOW_DETECTED_LINES:
            plot = cp.copy(self.map)
            for ln in lines:
                plot = cv2.line(plot, ln.pts[0].pos, ln.pts[1].pos, (0, 0, 255))
            cv2.imshow("Detected Lines", plot)

        # Merge overlapping lines
        merged = []
        while True:
            if len(lines) == 0:
                break
            lineA = lines.pop() # pick up one line from previous lines
            if len(merged) == 0: # no lines in merged
                merged.append(lineA)
                continue
            canMerge = False
            for lineB in merged:
                newLine, ok = lineA.merge(lineB) 
                if not ok: # can't merge, skip
                    continue
                canMerge = True
                merged.remove(lineB)
                merged.append(newLine)
            if not canMerge: # can't merge lineA with any line in merged, append it
                merged.append(lineA)
        
        # Visualize merged lines
        if SHOW_MERGED_LINES:
            plot = cp.copy(self.map)
            for ln in merged:
                plot = cv2.line(plot, ln.pts[0].pos, ln.pts[1].pos, (0, 0, 255))
            cv2.imshow("Merged Lines", plot)
        
        # Collect coordinates of all endpoints
        endpoints = []
        for ln in merged:
            for pt in ln.pts:
                endpoints.append((pt.pos, ln))

        # Find the line with closest endpoint to given point
        keyPts = [ROUTE_BEGIN_POSITION]
        while len(endpoints) > 0: # not all key points are collected
            lastPos = keyPts[-1]
            closestPt = min(endpoints, key=lambda pt: distSq(pt[0], lastPos))
            if len(endpoints) == 2 * len(merged): # begin endpoint of the whole path
                keyPts.append(closestPt[0])
            if distSq(closestPt[0], lastPos) > SEARCH_RANGE**2: 
                # closest point too far, end search
                break
            endpoints.remove(closestPt)
            line = closestPt[1]
            otherPos = line.otherEndpoint(closestPt[0])
            otherPosClosest = min(keyPts, key=lambda pt: distSq(pt, otherPos))
            endpoints.remove((otherPos, line))
            if distSq(otherPosClosest, otherPos) < CLOSEST_MIN_DIST**2: 
                # other endpoint too close to existing route points
                continue
            keyPts.append(otherPos)
        keyPts.pop(0)

        # Visualize generated key points
        if SHOW_KEYPOINTS:
            self.routePlot = cp.copy(self.map)
            for pos in keyPts:
                cv2.circle(self.routePlot, pos, 5, (0, 0, 255), thickness=-1)
            cv2.imshow("Key Positions", self.routePlot)

        return np.array(keyPts)

    def updateCar(self):
        # Denoise with gaussian blur
        blurred = cv2.GaussianBlur(self.map, (3, 3), 0)
        # Convert to HSV space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # Split channels
        hsvSplit = cv2.split(hsv)

        # Detect red region
        redMark = createMask(blurred, RED_HUE_RANGE, MARK_SAT_RANGE, MARK_VALUE_RANGE)
        # Find head position
        headPos = findMaskCenter(redMark)
        if SHOW_HEAD_MARK:
            cv2.imshow("Head Mark", redMark)
            print("Head position:", headPos)

        # Detect green region
        greenMark = createMask(blurred, GREEN_HUE_RANGE, MARK_SAT_RANGE, MARK_VALUE_RANGE)
        # Find tail position
        tailPos = findMaskCenter(greenMark)
        if SHOW_TAIL_MARK:
            cv2.imshow("Tail Mark", greenMark)
            print("Tail Position", tailPos)

        if SHOW_CAR_POS:
            plot = cp.copy(self.routePlot)
            cv2.circle(plot, headPos, 5, (0, 0, 0), thickness=-1)
            cv2.circle(plot, tailPos, 5, (0, 0, 0), thickness=-1)
            cv2.imshow("Car Position", plot)

        # Compute location and direction
        location = ((headPos[0] + tailPos[0])/2, (headPos[1] + tailPos[1] / 2))
        direction = (headPos[0] - tailPos[0], headPos[1] - tailPos[1])
        theta = np.arctan2(direction[1], direction[0])
        return np.array(location), theta


def distSq(pt1, pt2):
    return (pt1[0]-pt2[0])**2 + (pt1[1] - pt2[1])**2

def findMaskCenter(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    return findContourCenter(contours[0])

def findContourCenter(contour):
    M = cv2.moments(contour)
    return tuple([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

def createMask(img, hueRng, satRng, valRng):
    # Convert to HSV space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split channels
    hsvSplit = cv2.split(hsv)
    # Threshold in each channel
    hueMask = cv2.inRange(hsvSplit[0], hueRng[0], hueRng[1])
    satMask = cv2.inRange(hsvSplit[1], satRng[0], satRng[1])
    valMask = cv2.inRange(hsvSplit[2], valRng[0], valRng[1])
    return cv2.bitwise_and(cv2.bitwise_and(hueMask, satMask), valMask)


if __name__ == '__main__':
    route = RouteMap()
    route.capture(True)
    print(route.findRoute())
    while True:
        route.capture(True)
        print(route.updateCar())
        if cv2.waitKey(500) == 27:
            break
    cv2.waitKey(0)
