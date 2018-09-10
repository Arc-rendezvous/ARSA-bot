########################################################################
#
# Tugas Akhir EL
# 
# Gunawan Lumban Gaol
# Deskripsi : multi person pedestrian detection and tracking using 
# opencv builtin HOG
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
import KCF
import frame_convert2
import sys

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

	return (frame_out,centerxd,pick)

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
	flag_track2 = 0
	count = 0
	counttrack = 0
	prev_y_pixel = 0
	prev_x_pixel = 0
	tetaperpixel = 0.994837/400.0
	tracker = KCF.kcftracker(True, False, True, False)  # hog, fixed_window, multiscale, lab
	counttrack2 = 0
	prev_distance2 = 0
	# grab one frame at first to compare for background substraction
	frame,timestamp = freenect.sync_get_video()
	frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
	frame_resized = imutils.resize(frame, width=min(400, frame.shape[1]))
	frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
	print(frame_resized_grayscale.shape)
	# initialize centroid
	center = [[frame_resized.shape[1]/2, frame_resized.shape[0]/2]]
	center_fix = []
	# defining min cuoff area
	#min_area=(480/400)*frame_resized.shape[1] 
	min_area=(0.01)*frame_resized.shape[1] 
	print(frame_resized.shape) # (300,400,3)
	boxcolor=(0,255,0)
	timeout = 0; #variable for counting time elapsed
	key = ''
	temp = 1

	# save video
	countsave = 0
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
		#temp=background_subtraction(previous_frame, frame_resized_grayscale, min_area)

		# retrieve depth map
		depth,timestamp = freenect.sync_get_depth()
		depth = imutils.resize(depth, width=min(400, depth.shape[1]))
		print(depth.shape)
		depth2 = np.copy(depth)
		# orig = image.copy()
		if temp==1:
			if (flag_track2 == 0):
				frame_processed,center_fix,pick2 = detect_people(frame_resized,center,frame_resized,boxcolor)
				if (len(center_fix)>0):
					i = 0
					for b in center_fix:
				
						#print(b)
						#print("Point "+str(i)+": "+str(b[0])+" "+str(b[1]))
				

						x_pixel= b[1]
						y_pixel= b[0]
						print("x1:"+str(x_pixel)+"y1:"+str(y_pixel))
						rawDisparity = depth[(int)(x_pixel),(int)(y_pixel)]
						print("raw:"+str(rawDisparity))
						distance = 1/(-0.00307 * rawDisparity + 3.33)
						if (distance<0):
							distance = 0.5
						print ("Distance : " + str(distance))
						cv2.putText(frame_resized, "distance: {:.2f}".format(distance), (10, (frame_resized.shape[0]-(i+1)*25)-50), font, 0.65, (0, 0, 255), 3)
						cv2.putText(frame_resized, "Point "+str(i)+": "+str(b[0])+" "+str(b[1]), (10, frame_resized.shape[0]-(i+1)*25), font, 0.65, (0, 0, 255), 3)
						i = i + 1
					y_pix,x_pix = center_fix[0]
			
					endtime = time.time()
					#nucleo.write(("8,"+str(x_person)+","+str(y_person)).encode()) # send x_person and y_person
					if ((abs(prev_x_pixel-x_pix))<50 and (abs(prev_y_pixel-y_pix))<50):
						timeout = timeout + (endtime - starttime)
						if (timeout > 5):
							flag_track2 = 1;
							boxcolor = (255,0,0)
					else:
						timeout = 0
						boxcolor = (0,255,0)

					prev_y_pixel,prev_x_pixel = y_pix,x_pix
					# DEBUGGING #
					#print("Teta: " + str(teta) + "Distance: " + str(distance))
					print("Timeout: " + str(timeout))
					#print ("Distance : " + str(distance))
				elif(len(center_fix)<=0):
					timeout = 0
					boxcolor = (0,255,0)

			elif (flag_track2 == 1):
				if (counttrack2 == 0):
					iA,iB,iC,iD = pick2[0]
					
					# Draw new bounding box from body to only head figures
					
					tracker.init([iA,iB,iC-iA,iD-iB], frame_resized)
					counttrack2 = counttrack2 + 1
				elif (counttrack2 == 1):
					print(pick2[0])
					print("iA:"+str(iA)+"iB:"+str(iB)+"iC:"+str(iC)+"iD:"+str(iD))
					boundingbox = tracker.update(frame_resized)  #frame had better be contiguous
					boundingbox = list(map(int, boundingbox))
					cv2.rectangle(frame_resized,(boundingbox[0],boundingbox[1]), (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]), (255,0,0), 3)
					#GENERAL ASSUMPTION SINGLE PERSON TRACKING
					# start tracking...

					x_track = ((boundingbox[2])/2.0)+boundingbox[0]
					y_track = ((boundingbox[3])/2.0)+boundingbox[1]
					print("x:"+str(x_track)+"y:"+str(y_track))
					x_center = (frame_resized.shape[1]+1)/2
					y_center = (frame_resized.shape[0]+1)/2
					print(x_center,y_center)
					# compute teta asumsi distance lurus
				
					rawDisparity2 = depth2[(int)(y_track),(int)(x_track)]
					print("raw2:"+str(rawDisparity2))
					distance2 = 1/(-0.00307 * rawDisparity2 + 3.33)
					if (distance2<0):
						distance2 = prev_distance2					
					prev_distance2 = distance2
					
					#realx = (x_track-x_center)+(distance/30.0)
					#teta = math.atan(realx/distance) # if distance is tangensial
					#teta = math.asin((0.026458333*(x_track-x_center)/distance)) # if distance is euclidean
					teta = (x_track-x_center)*tetaperpixel
					print ("Teta: "+str(teta))
					print ("Distance2 : " + str(distance2))
					cv2.putText(frame_resized, "distance: {:.2f}".format(distance2), (10, (frame_resized.shape[0]-(i+1)*25)-50), font, 0.65, (0, 0, 255), 3)
					cv2.putText(frame_resized, "Point "+str(0)+": "+str(x_track)+" "+str(y_track), (10, frame_resized.shape[0]-(i+1)*25), font, 0.65, (0, 0, 255), 3)
					# send the teta and distance
					#nucleo.flush()
					#if(teta<0.0):
						#flag= nucleo.write(("7,"+format(teta,'1.2f')+","+format(distance2,'1.3f')).encode())
					#elif(teta>0.0):
						#flag= nucleo.write(("7,"+format(teta,'1.3f')+","+format(distance2,'1.3f')).encode())
					#print("WRITEIN1" + str(flag))
					print("Peak: "+str(tracker.getpeakvalue()))
					if(tracker.getpeakvalue()<0.6):
						counttrack2 = 0
						flag_track2 = 0
						#nucleo.flush()
						#nucleo.write("8,,,,,,,,,,,,".encode())
						print("WRITEOUT")

			#frame_resized = cv2.flip(frame_resized, 0)
			cv2.imshow("Detected Human", frame_resized)
			cv2.imshow("Depth", frame_convert2.pretty_depth_cv(depth))
			#cv2.imshow("Depth2", frame_convert2.pretty_depth_cv(depth2))
			# cv2.imshow("Original", frame)
		else:
			count=count+1
			print("Number of frame skipped in the video= " + str(count))

		# compute the fps
		fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
		print("FPS: " + str(fps))
		#outframe = open("/home/ubuntu/Progress\ TA/Integrasi/rgb640/%d.jpg" % countsave, 'wb+')
		cv2.imwrite('%d.jpg' % countsave, frame_resized) # Save image...
		countsave = countsave + 1
		key = cv2.waitKey(5)

	cv2.destroyAllWindows()
	freenect.sync_stop()
	nucleo.close()
	print("\nFINISH")

if __name__ == "__main__":
	global distance
	global teta
	global depth
	global pick2, fps
	# Setup the serial port
	#nucleo = serial.Serial()
	#nucleo.baudrate = 115200
	#nucleo.port = '/dev/ttyACM0'
	#nucleo.close()
	#nucleo.bytesize=serial.EIGHTBITS
	#nucleo.timeout = 1
	#nucleo.open()
	#nucleo.flush()
	#print("connected to: " + nucleo.portstr)
	time.sleep(3) # give time to wake up
	subject_label = 1
	font = cv2.FONT_HERSHEY_SIMPLEX
	# initialize the HOG descriptor/person detector
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	main()
