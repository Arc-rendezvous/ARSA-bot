########################################################################
#
# Tugas Akhir EL
# 
# Gunawan Lumban Gaol
# Deskripsi : multi person human detection using opencv builtin HOG
# Kinect 360, Python 3+, dan OpenCV 3.2.0.
#
#  ___________640px___________
# |                           |
# |                           |
# |                           480px
# |                           |
# |___________________________|
# 
#
########################################################################

#!/usr/bin/env python
# import the necessary packages
from imutils.object_detection import non_max_suppression	
import numpy as np
import imutils
import cv2
import freenect
import time
import os
import glob
import serial
import struct
import math

def detect_people(frame,center,frame_out,bboxcolor = (0,255,0)):
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
		cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	idx = 0
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(frame_out, (xA, yA), (xB, yB), bboxcolor, 2)
		cv2.putText(frame_out,'Person '+str(idx),(xA,yA-10),0,0.3,bboxcolor)
		idx = idx + 1
		# calculate the center of the object
		centerxd.append([(xA+xB)/2,(yA+yB)/2])

	return (frame_out,centerxd)

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

def encodex(x_pixel):
	if(x_pixel<130):
		x_norm = 1
	elif(x_pixel>230):
		x_norm = 2
	else:
		x_norm = 0

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
	print(frame_resized.shape[1])
	boxcolor=(0,255,0)
	timeout = 0; #variable for counting time elapsed
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
			frame_processed,center_fix = detect_people(frame_resized_grayscale,center,frame_resized,boxcolor)
			if (len(center_fix)>0):
				xnorm = encodex(center_fix[0][0]) # retrieve coded position from image
				prev_xnorm = xnorm;
				#nucleo.write(struct.pack('>B',xnorm))
				endtime = time.time()
				if (prev_xnorm == xnorm):
					timeout = timeout + (endtime - starttime)
					if (timeout > 10):
						boxcolor = (255,0,0)
				else:
					timeout = 0
					boxcolor = (0,255,0)
				print("X_NORM: " + str(xnorm))
				print("Timeout: " + str(timeout))
				rawDisparity = depth[(int)(center_fix[0][1]),(int)(center_fix[0][0])]
				distance = 100/(-0.00307 * rawDisparity + 3.33)
				cv2.putText(frame_resized, "distance: {:.2f}".format(distance), (10, frame_processed.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
				#print("Distance: " + str(depth.shape) + str(frame_processed.shape))
			else:
				timeout = 0
				boxcolor = (0,255,0)
			i = 0
			for b in center_fix:
				cv2.putText(frame_resized, "Point "+str(i)+": "+str(b[0])+" "+str(b[1]), (10, frame_resized.shape[0]-(i+1)*35), font, 0.65, (0, 0, 255), 3)
				#print(b)
				#print("Point "+str(i)+": "+str(b[0])+" "+str(b[1]))
				i = i + 1
			#frame_resized = cv2.flip(frame_resized, 0)
			cv2.imshow("Detected Human", frame_resized)
			#cv2.imshow("Depth", depth)			
			# cv2.imshow("Original", frame)
		else:
			count=count+1
			print("Number of frame skipped in the video= " + str(count))

		# compute the fps
		fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
		print("FPS: " + str(fps))
		key = cv2.waitKey(5)

	cv2.destroyAllWindows()
	freenect.sync_stop()
	nucleo.close()
	print("\nFINISH")

if __name__ == "__main__":
	"""	
	# Setup the serial port
	nucleo = serial.Serial()
	nucleo.baudrate = 57600
	nucleo.port = '/dev/ttyACM0'
	nucleo.close()
	#nucleo.parity=serial.PARITY_ODD
	#nucleo.stopbits=serial.STOPBITS_ONE
	nucleo.bytesize=serial.EIGHTBITS
	nucleo.timeout = 1
	nucleo.open()
	nucleo.flush()	
	print("connected to: " + nucleo.portstr)
	time.sleep(3) # give time to wake up
	"""
	subject_label = 1
	font = cv2.FONT_HERSHEY_SIMPLEX
	# initialize the HOG descriptor/person detector
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	main()
