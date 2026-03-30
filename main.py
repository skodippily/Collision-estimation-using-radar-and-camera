import threading
import time
import numpy as np

from AWR1843_Read_Data import readData_AWR1843 as radar


class RadarReading:
    def __init__(self):
        # Create a stop flag for safe shutdown
        self.stop_event = threading.Event()

    def upateRadarData(self):
        while not self.stop_event.is_set():
            radar.updateFromMain()
            time.sleep(0.1)

    def dataProcess(self):
        while not self.stop_event.is_set():
            print(f"Test dict####={radar.getData()}")
            time.sleep(0.05)

    def main(self):
        radar.initRadar()
        time.sleep(2)
        # Create threads
        radarThread = threading.Thread(target=self.upateRadarData)
        processingThread = threading.Thread(target=self.dataProcess)

        # Start threads
        # radarThread.start()
        # processingThread.start()

        print("Threads started. Press Ctrl+C to stop.")

        # Keep main thread alive
        try:
            while not self.stop_event.is_set():
                radar.updateFromMain()
                print(f"Test dict####={radar.getData()}")
                radar.updatePlot()
        except KeyboardInterrupt:
            print("\nStopping threads...")
            self.stop_event.set()

        # Wait for threads to finish
        radarThread.join()
        processingThread.join()
        print("All threads closed safely.")


if __name__ == "__main__":
    try:
        rr = RadarReading()
        rr.main()
        # radar.test()
        # AWR1843 = AWR1843_Read_Data
        # AWR1843.test()
        pass

    except KeyboardInterrupt:
        rr.stop_event.set()
        radar.closePortsAndPlot()
        print("\nClosing...")



