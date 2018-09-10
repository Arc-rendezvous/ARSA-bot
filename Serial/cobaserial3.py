############################################################
# 
# Testing serial between nucelo f446re and nvidia jetson TK1
#
############################################################
# Syarat bisa harus konek muncul as usb nucleo di jetsonnya



def main():
	x = 1.2345
	y= 5.678
	
	abcd = "9,"+format(x,'.4f')+","+format(y,'.4f')
	print(abcd)
#	print (Decimal(1.2))

if __name__ == "__main__":
	main()
