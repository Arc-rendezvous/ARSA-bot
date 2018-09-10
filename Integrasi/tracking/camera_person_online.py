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

class Camera_person_online():
	def get_frame(self, countif):
		global previous_frame 
		global min_area
		global center
		global count
		global boxcolor
		global hog	
		global frame_processed
		global timeout
		global font
		global center_fix
		global flag_track
		global prev_x_pixel
		global prev_y_pixel
		global counttrack
		global tetaperpixel
		global nucleo
		global distance
		global teta
		global i
		global b
		global flag
		global prev_distance
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

		def encodex(x_pixel):
			if(x_pixel<130):
				x_norm = 1
			elif(x_pixel>230):
				x_norm = 2
			else:
				x_norm = 0

			return x_norm

		if (countif < 1):
			# setup the serial port
			nucleo = serial.Serial()
			nucleo.port = '/dev/ttyACM0'
			nucleo.baud = 57600
			nucleo.close()
			nucleo.open()
			nucleo.flush()
			time.sleep(2)
			print("connected to: " + nucleo.portstr)
			print("Running...");
			subject_label = 1
			font = cv2.FONT_HERSHEY_SIMPLEX
			# initialize the HOG descriptor/person detector
			hog = cv2.HOGDescriptor()
			hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
			time.sleep(1)

			frame = np.zeros((480,640,3), np.uint8)
			flag_track = 0
			count = 0
			counttrack = 0
			countsave = 0
			prev_y_pixel = 0
			prev_x_pixel = 0
			tetaperpixel = 0.3167/400.0
			# grab one frame at first to compare for background substraction
			frame,timestamp = freenect.sync_get_video()
			#time.sleep(5)
			frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
			frame_resized = imutils.resize(frame, width=min(400, frame.shape[1]))
			frame_resized_grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

			# initialize centroid
			center = [[frame_resized.shape[1]/2, frame_resized.shape[0]/2]]
			center_fix = []
			# defining min cuoff area
			#min_area=(480/400)*frame_resized.shape[1] 
			min_area=(0.01)*frame_resized.shape[1] 
			boxcolor=(0,255,0)
			timeout = 0; #variable for counting time elapsed
			temp=1;
			
			previous_frame = frame_resized_grayscale
			# retrieve new RGB frame image
			# Frame generation for Browser streaming with Flask...
			self.outframe = open("stream.jpg", 'wb+')
			cv2.imwrite("stream.jpg", frame) # Save image...
		
			return self.outframe.read()
		else:
			# start timer
			timer = cv2.getTickCount()
			starttime = time.time()
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
			
					i = 0
					for b in center_fix:
					
						#print(b)
						#print("Point "+str(i)+": "+str(b[0])+" "+str(b[1]))
					

						x_pixel= b[1]
						y_pixel= b[0]
						rawDisparity = depth[(int)(x_pixel),(int)(y_pixel)]
						distance = 1/(-0.00307 * rawDisparity + 3.33)
						if (distance<0):
							distance = 0.5
						print ("Distance : " + str(distance))
						cv2.putText(frame_resized, "distance: {:.2f}".format(distance), (10, (frame_resized.shape[0]-(i+1)*25)-50), font, 0.65, (0, 0, 255), 3)
						cv2.putText(frame_resized, "Point "+str(i)+": "+str(b[0])+" "+str(b[1]), (10, frame_resized.shape[0]-(i+1)*25), font, 0.65, (0, 0, 255), 3)
						i = i + 1
					y_pix,x_pix = center_fix[0]
				
					endtime = time.time()
					if (flag_track == 0):
						nucleo.flush()
						nucleo.write("8,,,,,,,,,,,,".encode())
						if ((abs(prev_x_pixel-x_pix))<50 and (abs(prev_y_pixel-y_pix))<50):
							timeout = timeout + (endtime - starttime)
							if (timeout > 5):
								flag_track = 1;
								boxcolor = (255,0,0)
						else:
							timeout = 0
							boxcolor = (0,255,0)
					elif (flag_track == 1):
						#GENERAL ASSUMPTION SINGLE PERSON TRACKING
						# start tracking...
						#nucleo.write("9,,,,,,,,,,,,".encode())
					
						x_track,y_track = x_pix,y_pix
						x_center = (frame_resized.shape[1]+1)/2
						y_center = (frame_resized.shape[0]+1)/2
						print(x_center,y_center)
						# compute teta asumsi distance lurus
					
						rawDisparity = depth[(int)(x_track),(int)(y_track)]
						distance = 1/(-0.00307 * rawDisparity + 3.33)
						if (distance<0):
							distance = prev_distance
						prev_distance = distance
						#realx = (x_track-x_center)+(distance/30.0)
						#teta = math.atan(realx/distance) # if distance is tangensial
						#teta = math.asin((0.026458333*(x_track-x_center)/distance)) # if distance is euclidean
						teta = (y_track-x_center)*tetaperpixel
						print("teta="+str(teta)+"x:"+str(x_track)+"y:"+str(y_track))
					
						# send the teta and distance
						nucleo.flush()
						if(teta<0.0):
							flag= nucleo.write(("7,"+format(teta,'1.2f')+","+format(distance,'1.3f')).encode())
						elif(teta>0.0):
							flag= nucleo.write(("7,"+format(teta,'1.3f')+","+format(distance,'1.3f')).encode())
						#print("WRITEIN1" + str(flag))

					prev_y_pixel,prev_x_pixel = y_pix,x_pix
					# DEBUGGING #
					#print("Teta: " + str(teta) + "Distance: " + str(distance))
					print("Timeout: " + str(timeout))
					print ("Distance : " + str(distance))
				elif(len(center_fix)<=0):
					# count how many frame skipped without people detected while tracking
					if (flag_track==1):
						counttrack = counttrack + 1
						# continue tracking after a maximum of 50 frame without people taking previous location as target
						if (counttrack<=20):
							# still print the last condition to the output image
							cv2.putText(frame_resized, "distance: {:.2f}".format(distance), (10, (frame_resized.shape[0]-25)-50), font, 0.65, (0, 255, 255), 3)
							cv2.putText(frame_resized, "Point "+str(i)+": "+str(b[0])+" "+str(b[1]), (10, frame_resized.shape[0]-25), font, 0.65, (0, 255, 255), 3)
							# still send the nucleo tracking state
							nucleo.flush()
							if(teta<0.0):
								flag= nucleo.write(("7,"+format(teta,'1.2f')+","+format(distance,'1.3f')).encode())
							elif(teta>0.0):
								flag= nucleo.write(("7,"+format(teta,'1.3f')+","+format(distance,'1.3f')).encode())
							print("WRITEIN" + str(flag))
						elif (counttrack>20):
							counttrack = 0
							flag_track = 0
							# tell nucleo to leave tracking state
							nucleo.flush()
							nucleo.write("8,,,,,,,,,,,,".encode())
							print("WRITEOUT")
					elif (flag_track==0):
						timeout = 0
						boxcolor = (0,255,0)
						nucleo.flush()
						nucleo.write("8,,,,,,,,,,,,".encode())

			
				#frame_resized = cv2.flip(frame_resized, 0)
				#cv2.imshow("Detected Human", frame_resized)
				#cv2.imshow("Depth", depth)			
				# cv2.imshow("Original", frame)
			else:
				count=count+1
				#print("Number of frame skipped in the video= " + str(count))

			# compute the fps
			fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

			#print("FPS: " + str(fps))
		
			# Frame generation for Browser streaming with Flask...
			self.outframe = open("stream.jpg", 'wb+')
			cv2.imwrite("stream.jpg", frame_resized) # Save image...
			cv2.imwrite('web_%d.jpg' % countsave, frame_resized) # Save image...
			countsave = countsave + 1
			return self.outframe.read()
		
