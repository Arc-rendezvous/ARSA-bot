############################################################
# 
# Testing serial between nucelo f446re and nvidia jetson TK1
#
############################################################
# Syarat bisa harus konek muncul as usb nucleo di jetsonnya
import serial
import struct
import time
import io

buffer_length = 11

def main():
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
	x = 220 
	y = 0
	time.sleep(3)

	# entering tracking state
	#nucleo.write("9,,,,,,,,,,,,".encode())
	
	
	#nucleo.write((str(x)+","+str(y)).encode())

	#hello = nucleo.readline()
	#print(hello)
	
	# leaving tracking state tell nucleo
	#nucleo.write("8,,,,,,,,,,,,".encode())

	#hello = nucleo.readline()
	#print(hello)
	while(True):
		nucleo.write(("7,"+format(x,'3.1f')+","+format(y,'1.3f')).encode())
	
		#hello = nucleo.readline()
		#print(hello)
		time.sleep(0.5)
	#nucleo.write("hello      ".encode())
	#hello = nucleo.readline()
	#print(hello)

if __name__ == "__main__":
	main()
