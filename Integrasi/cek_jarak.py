########################################################################
#
# Tugas Akhir EL
# 
# Gunawan Lumban Gaol
# Deskripsi : multi person human detection using opencv builtin HOG
# Kinect 360, Python 3+, dan OpenCV 3.2.0.
#
########################################################################

#!/usr/bin/env python
# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import freenect
import frame_convert2
import time
import os
import glob
import serial
import struct

def detect_people(frame,center):
	"""
	detect humans using HOG descriptor
	Args:
		frame:
	Returns:
		processed frame, center of every bb box
	"""
	centerxd = []
	(rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.06)
	rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	idx = 0
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
		cv2.putText(frame,'Person '+str(idx),(xA,yA-10),0,0.3,(0,255,0))
		idx = idx + 1
		# calculate the center of the object
		centerxd.append([(xA+xB)/2,(yA+yB)/2])

	return (frame,centerxd)

def background_subtraction(previous_frame, frame_resized_grayscale, min_area):
	"""
	This function returns 1 for the frames in which the area 
	after subtraction with previous frame is greater than minimum area
	defined. 
	Thus expensive computation of human detection face detection 
	and face recognition is not done on all the frames.
	Only the frames undergoing significant amount of change (which is controlled min_area)
	are processed for detection and recognition.
	"""
	frameDelta = cv2.absdiff(previous_frame, frame_resized_grayscale)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	temp=0
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) > min_area:
			temp=1
	return temp		

def sercomm(x_pixel):
	if(x_pixel<130):
		x_norm = 1
	elif(x_pixel>230):
		x_norm = 2
	else:
		x_norm = 0

	nucleo.write(struct.pack('>B',x_norm))
#	nucleo.write(x).encode('utf-8'));
#	while True:
#		if (nucleo.inWaiting()>0):
#			line = nucleo.readline()
#			print (line)
	
	return x_norm
def main():
	print("Running...");

	count = 0
	# grab one frame at first to compare for background substraction
	frame,timestamp = freenect.sync_get_video()
	frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
	frame_resized = imutils.resize(frame, width=min(400, frame.shape[1]))
	frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

	# initialize centroid
	center = [[frame_resized.shape[1]/2, frame_resized.shape[0]/2]]
	center_fix = []
	# defining min cuoff area
	min_area=(480/400)*frame_resized.shape[1] 

	key = ''
	while key != 113:  # for 'q' key
		# start timer
		timer = cv2.getTickCount()
		starttime = time.time()

		previous_frame = frame_resized_grayscale
		# retrieve new RGB frame image
		frame,timestamp = freenect.sync_get_video()
		frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
		frame_resized = imutils.resize(frame, width=min(400, frame.shape[1]))
		frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
		temp=background_subtraction(previous_frame, frame_resized_grayscale, min_area)

		# retrieve depth map
		depth,timestamp = freenect.sync_get_depth()
		depth = imutils.resize(depth, width=min(400, depth.shape[1]))
		# orig = image.copy()
		if temp==1:		
			frame_processed,center_fix = detect_people(frame_resized,center)
			if (len(center_fix)>0):
				#xnorm = sercomm(center_fix[0][0]) # send x_norm to nucleo
				#print("X_NORM: " + str(xnorm))	
				distance = depth[(int)(center_fix[0][1]),(int)(center_fix[0][0])]
				cv2.putText(frame_processed, "distance: {:.2f}".format(distance), (10, frame_processed.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
				#print("Distance: " + str(depth.shape) + str(frame_processed.shape))
			i = 0
			for b in center_fix:
				cv2.putText(frame_processed, "Point "+str(i)+": "+str(b[0])+" "+str(b[1]), (10, frame_processed.shape[0]-(i+1)*35), font, 0.65, (0, 0, 255), 3)
				#print(b)
				#print("Point "+str(i)+": "+str(b[0])+" "+str(b[1]))
				i = i + 1
			cv2.imshow("Detected Human", frame_processed)
			#cv2.imshow("Depth", depth)			
			# cv2.imshow("Original", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("b"):
				break
			endtime = time.time()
			print("Time to process a frame: " + str(starttime-endtime))
		else:
			count=count+1
			print("Number of frame skipped in the video= " + str(count))

		# cv2.putText(depth, "distance: {:.2f}".format(distance), (10, depth.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)

		# compute the fps
		fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

		print("FPS: " + str(fps))

		key = cv2.waitKey(5)

	cv2.destroyAllWindows()
	print("\nFINISH")

if __name__ == "__main__":
	# setup the serial port
	#nucleo = serial.Serial('/dev/ttyACM0', 115200)
	#nucleo.flushInput()
	#time.sleep(2)
	#print("connected to: " + nucleo.portstr)
	subject_label = 1
	font = cv2.FONT_HERSHEY_SIMPLEX
	# initialize the HOG descriptor/person detector
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	main()
