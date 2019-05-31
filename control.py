import serial
import time
import numpy as np
import cv2
from serial.tools import list_ports
from routemap import RouteMap

# Threshold to decide whether the car has reached key point
DIST_THRESH = 20
# Threshold to decide whether car should move forward
ANGLE_THRESH = 0.25
# Period of updating perspective matrix in route map
MAT_UPDATE_PERIOD = 3
# Period (seconds) between two control signals
SLEEP_SECONDS = 1

class Controller:
	def __init__(self, routemap, ser):
		self.map = routemap
		self.ser = ser

	def run(self):
		# Intialize route map data
		self.map.capture(True):
		route = list(self.map.findRoute())
		print("Route points:", route)
		targPos = route.pop(0)
		print("Target position:", targPos)

		# Running loop
		nFrames = 1
		while True:
			self.map.capture(nFrames % MAT_UPDATE_PERIOD == 0)
			nFrames += 1
			carPos, carDir = self.map.updateCar()
			print("Car position:", carPos, "direction:", carDir)
			if isNear(carPos, targPos):
				if len(route) == 0: # no remaining route points
					self.ser.write('P'.encode('ascii')) # park here
					break
				targPos = route.pop(0)
				print("Target position:", targPos)
			self._move(targPos, carPos, carDir)
			if cv2.waitKey(SLEEP_SECONDS * 1000) == 27:
				break

		cv2.destroyAllWindows()


	def _move(self, targPos, carPos, carDir):
		targVec = targPos - carPos
		targDir = np.arctan2(targVec[1], targVec[0])
		dAngle = targDir - carDir
		if np.abs(dAngle) < ANGLE_THRESH:
			self.ser.write('A'.encode('ascii')) # move ahead
		elif dAngle > 0 and dAngle < np.pi:
			self.ser.write('S'.encode('ascii')) # turn right
		else:
			self.ser.write('M'.encode('ascii')) # turn left

def isNear(pt1, pt2):
	return np.linalg.norm(pt1 - pt2) < DIST_THRESH


if __name__ == '__main__':
	ser = serial.Serial('COM3', 9600, timeout=0.5)
	rtmap = RouteMap()
	ctrl = Controller(rtmap, ser)
	ctrl.run()
	