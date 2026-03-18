# Date: 16-03-2026
# Author: Sachi(u3277899)

# Acknowledged these projects
# https://github.com/G20Michu/RdLib
# https://github.com/valeriofantozzi/Ai-Thinker-RD-03-ESP32/blob/main/ESP32_RD03D/ESP32_RD03D.ino
# https://github.com/javier-fg/Arduino_RD-03D

import serial
import math
import time


class TargetData:
    def __int__(self):
        pass

    def clearValues(self):
        self.id = 0                 # mm
        self.x = 0                  # mm
        self.y = 0                  # mm
        self.speed = 0              # cm/s
        self.distanceRes = 0        # mm
        self.distance = 0
        self.angle = 0
        self.isValid = False

    def setValues(self, x, y, speed, distanceRes):
        # If distanceResolution is zero, the measurement is not valid
        # if (distanceRes == 0):
        #     self.clearValues()
        #     return 0
        self.x = x
        self.y = y
        self.speed = speed
        self.distanceRes = distanceRes

        # Compute distance and angle based on x and y values.
        self.distance = math.sqrt(x**2 + y**2)
        self.angle = math.degrees(math.atan2(x, y))

        # If distance is more than theoretical, measurement not valid
        # if (self.distance > self.MAX_DISTANCE):
        #     self.clearValues()
        #     return 0
        self.isValid = True
        return 1

    def printInfo(self):
        # print(f"Target-{self.id}:")

        # if (not self.isValid):
        #     print("-- no valid data --")
        #     return 0

        print(f"distance: {self.distance/10.0} cm")
        print(f"angle: {self.angle}")
        print(f"x: {self.x}")
        print(f"y: {self.y}")
        print(f"speed: {self.speed}cm/s")


class RD03D:
    def __init__(self, comPort, baudRate=256000, bufferSize=1000):
        try:
            self.ser = serial.Serial(comPort, baudRate, 8, 'N', 1)
        except serial.SerialException:
            return None
        except ValueError:
            # Invalid port parameters
            return None
        print("Radar connected")

        self.MAX_TARGETS = 3
        self.DETECT_MULTI_TARGET = 0
        self.TIMEOUT = 500
        self.CMD_TARGET_DETECTION_SINGLE = [
            0xFD, 0xFC, 0xFB, 0xFA,
            0x02, 0x00, 0x80, 0x00,
            0x04, 0x03, 0x02, 0x01
        ]

        self.CMD_TARGET_DETECTION_MULTI = [
            0xFD, 0xFC, 0xFB, 0xFA,
            0x02, 0x00, 0x90, 0x00,
            0x04, 0x03, 0x02, 0x01
        ]
        self.bufferSize = bufferSize
        self._bufferRx = bytearray(bufferSize)
        self._bufferTx = bytearray(30)

        self.targets = [TargetData() for _ in range(self.MAX_TARGETS)]
        for i in range(self.MAX_TARGETS):
            self.targets[i].idNum = i + 1

    def close(self):
        self.ser.close()

    def readData(self):
        """Read and parse a complete data frame from the radar"""
        frame_data = bytearray()
        header_found = False

        while True:
            if self.ser.in_waiting:
                byte = ord(self.ser.read())

                if not header_found:
                    frame_data.append(byte)
                    # Check for header sequence
                    if len(frame_data) >= 4:
                        if (frame_data[-4:] == bytes([0xAA, 0xFF, 0x03, 0x00])):
                            header_found = True
                            # Keep only the header
                            frame_data = frame_data[-4:]
                elif header_found:
                    frame_data.append(byte)

                    # Check if we have a complete frame
                    if len(frame_data) >= (4 + 24 + 2):  # Header + 3*8 bytes data + Footer
                        if frame_data[-2:] == bytes([0x55, 0xCC]):
                            # Valid frame received, parse targets
                            data_start = 4  # After header

                            for i in range(self.MAX_TARGETS):  # 3 possible targets
                                target_data = frame_data[data_start +
                                                         i*8:data_start + (i+1)*8]
                                self._parse_target_data(i, target_data)

                        else:
                            # Invalid frame, start over
                            frame_data = bytearray()
                            header_found = False

    def _decode_raw(self, value: int) -> float:
        """Decode a coordinate value according to the protocol specification"""
        # Check if highest bit is set (positive/negative indicator)
        is_negative = not bool(value & 0x8000)
        # Get absolute value (15 bits)
        abs_value = value & 0x7FFF
        return -abs_value if is_negative else abs_value

    def _parse_target_data(self, i, data: bytes):
        """Parse 8 bytes of target data into a RadarTarget object"""
        if all(b == 0 for b in data):  # Check if target data is all zeros
            return None

        # Extract values (little endian)
        x_raw = int.from_bytes(data[0:2], byteorder='little')
        y_raw = int.from_bytes(data[2:4], byteorder='little')
        speed_raw = int.from_bytes(data[4:6], byteorder='little')
        distance = int.from_bytes(data[6:8], byteorder='little')

        __res = self.targets[i].setValues(self._decode_raw(x_raw),
                                          self._decode_raw(y_raw),
                                          self._decode_raw(speed_raw),
                                          self._decode_raw(distance))
        self.targets[i].printInfo()

        if not __res:
            print("Error reading data")

    def set_multi_mode(self, multi_mode=True):
        """Set Radar mode: True=Multi-target, False=Single-target"""
        MULTI_TARGET_CMD = bytes(
            [0xFD, 0xFC, 0xFB, 0xFA, 0x02, 0x00, 0x90, 0x00, 0x04, 0x03, 0x02, 0x01])

        cmd = MULTI_TARGET_CMD if multi_mode else self.SINGLE_TARGET_CMD
        self.ser.write(cmd)
        self.ser.read()

    def getValue(self):
        print('get value')
        for target in self.targets:
            target.printInfo()


def main():
    sensor1 = RD03D(comPort='COM6')
    sensor1.set_multi_mode()
    # sensor1.initialize()
    try:
        print("Reading radar data...")
        while True:
            sensor1.readData()
            # sensor1.getValue()

    except KeyboardInterrupt:
        print("\nClosing serial port...")
        sensor1.close()


if __name__ == "__main__":
    main()
