import time
from smbus2 import SMBus

# BH1750 address
DEVICE_ADDR = 0x23

# Mode for continuous high-resolution measurement (1 lx resolution)
CONTINUOUS_HIGH_RES_MODE = 0x10

def read_light():
    with SMBus(1) as bus:
        data = bus.read_i2c_block_data(DEVICE_ADDR, CONTINUOUS_HIGH_RES_MODE, 2)
        lux = ((data[0] << 8) + data[1]) / 1.2
        return lux

if __name__ == "__main__":
    try:
        while True:
            lux = read_light()
            print(f"Light Intensity: {lux:.2f} lx")
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nMeasurement stopped by user.")
