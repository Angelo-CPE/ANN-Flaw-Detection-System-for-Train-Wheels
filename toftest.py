import VL53L0X
import time
import keyboard  # Make sure to install it with: pip install keyboard

def main():
    tof = VL53L0X.VL53L0X()
    tof.open()
    tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.GOOD)

    print("Press 'q' to quit...\n")

    try:
        while True:
            distance = tof.get_distance()
            print("Distance: {} mm".format(distance))
            time.sleep(0.2)  # Delay to avoid spamming output
            if keyboard.is_pressed('q'):
                print("Exiting...")
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        tof.stop_ranging()
        tof.close()

if __name__ == "__main__":
    main()

