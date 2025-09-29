import serial, time

ser = serial.Serial('/dev/ttyAMA3', 115200, bytesize=8, parity='N', stopbits=1, timeout=1)

frame = bytes(b'ABCD')
ser.write(frame)
time.sleep(0.001)
ser.write(b'Hello\n')
ser.close()

