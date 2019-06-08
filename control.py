import time

import cv2
import numpy as np
import serial
from route import RouteMap
from geometry import *

# Threshold to decide whether the car has reached key point
DIST_THRESH = 20
# Threshold to decide whether car should move forward
ANGLE_THRESH = 0.3
# Period (seconds) between two control signals
SLEEP_SECONDS = 500
# Period of updating perspective matrix in route map
MAT_UPDATE_PERIOD = 2

# Command character of car control
# Must convert unicode string to bytes before writing to serial
AHEAD = 'A'.encode("ascii")
PARK = 'P'.encode("ascii")
LEFT = 'M'.encode("ascii")
RIGHT = 'S'.encode("ascii")

class Controller:
	def __init__(self, routemap: RouteMap, ser: serial.Serial):
		self.map = routemap
		self.ser = ser

	def run(self):
		# Intialize route map data
		self.map.capture(True)
		route = list(self.map.findRoute())
		print("Route points:", route)
		targPos = route.pop(0)
		print("Target position:", targPos)

		# Running loop
		nFrames = 1
		while True:
			# Capture new frame
			self.map.capture(nFrames % MAT_UPDATE_PERIOD == 0)
			nFrames += 1
			# Update car position
			carPos, carDir = self.map.updateCar()
			print("Car position:", carPos, "direction:", carDir)
			print("Distance to next point:", dist(carPos, targPos))
			# Check if target position is reached
			if isNear(carPos, targPos):
				if len(route) == 0: # no remaining route points
					self.ser.write(PARK) # park here
					break
				targPos = route.pop(0)
				print("Target position:", targPos)
			# Plot car position and route points
			plot = self.map.carPlot
			for pos in route:
				cv2.circle(plot, pos, 5, (255, 0, 0), thickness=-1)
			cv2.circle(plot, targPos, 5, (0, 255, 0), thickness=-1)
			cv2.imshow("Car position", plot)
			# Send instruction to car
			self._move(targPos, carPos, carDir)
			if cv2.waitKey(SLEEP_SECONDS) == 27:
				break
		
		# Finish work
		cv2.destroyAllWindows()
		print("Finished")

	def _move(self, targPos, carPos, carDir):
		targVec = targPos - carPos
		targDir = np.arctan2(targVec[1], targVec[0])
		dAngle = targDir - carDir
		if np.abs(dAngle) < ANGLE_THRESH:
			print("AHEAD")
			self.ser.write(AHEAD) # move ahead
		elif dAngle > 0 and dAngle < np.pi:
			print("TURN RIGHT")
			self.ser.write(RIGHT) # turn right
		else:
			print("TURN LEFT")
			self.ser.write(LEFT) # turn left

def testCommand(ser):
	command = [PARK, AHEAD, LEFT, AHEAD, RIGHT, AHEAD, PARK]
	for c in command:
		ser.write(c)
		time.sleep(1)

def main():
	# Intialize bluetooth serial port
	ser = serial.Serial('COM3', 9600, timeout=0.5)
	# Test serial command
	testCommand(ser)
	# Create route map and controller
	rtmap = RouteMap()
	ctrl = Controller(rtmap, ser)
	# Main running function
	ctrl.run()

if __name__ == '__main__':
	main()
