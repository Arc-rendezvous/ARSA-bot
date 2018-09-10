############################################################
# 
# Testing serial between nucelo f446re and nvidia jetson TK1
#
############################################################

import serial
import struct
import time

def main():
	nucleo = serial.Serial()
	nucleo.close()
	nucleo.baudrate = 57600
	nucleo.port = '/dev/ttyACM0'
	nucleo.timeout = 1
	nucleo.open()
	nucleo.flushInput()
	print("connected to: " + nucleo.portstr)
	
	x = 5
	x_str = "12345"
	time.sleep(5)
	nucleo.write(x_str.encode())
#	nucleo.write(x.to_bytes(2,'big',signed=True))
#	print(x.to_bytes(2,'big',signed=True))
#	nucleo.write(struct.pack('>BB',x,y))
	while True:
		if (nucleo.inWaiting()>0):
			line = nucleo.read(5)
			#line = ((((nucleo.readline()).decode()).rstrip()).split(','))
			#hilangkan \n
			print(line)
			#print(line[0])
			#print(float(line[0]) +1)
			#print(float(line[1])+1)
			#print(float(line[2])+1)
#			line[0] = float(line[0]) + 1
			
#			print (line)
				
if __name__ == "__main__":
	main()

	


