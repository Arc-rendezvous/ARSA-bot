import subprocess
import sys

if (len(sys.argv) == 2):
	if(sys.argv[1] == "main"):
		subprocess.call(['./MainProgram.sh'])
	elif(sys.argv[1] == "patrol"):
		subprocess.call(['./Patrol.sh'])
	elif(sys.argv[1] == "track"):
		subprocess.call(['./Tracking.sh'])
	elif(sys.argv[1] == "trackpatrol"):
		subprocess.call(['./TrackingPatrol.sh'])
elif (len(sys.argv) == 1):
	print("Not enough argument, required 2 arguments.")
	print("List of Arguments: ")
	print("1. main")
	print("2. patrol")
	print("3. track")
	print("4. trackpatrol")
