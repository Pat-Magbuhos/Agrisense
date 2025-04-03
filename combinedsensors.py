import time
import glob
import board
import adafruit_dht
from smbus2 import SMBus

# --- DS18B20 Setup ---
base_dir = '/sys/bus/w1/devices/'
device_folder = glob.glob(base_dir + '28*')[0]
device_file = device_folder + '/w1_slave'

def read_temp_raw():
    with open(device_file, 'r') as f:
        lines = f.readlines()
    return lines

def read_ds18b20_temp():
    lines = read_temp_raw()
    while lines[0].strip()[-3:] != 'YES':
        time.sleep(0.2)
        lines = read_temp_raw()
    temp_pos = lines[1].find('t=')
    if temp_pos != -1:
        temp_string = lines[1][temp_pos+2:]
        temp_c = float(temp_string) / 1000.0
        return temp_c

# --- BH1750 Setup ---
DEVICE_ADDR = 0x23
CONTINUOUS_HIGH_RES_MODE = 0x10

def read_light():
    with SMBus(1) as bus:
        data = bus.read_i2c_block_data(DEVICE_ADDR, CONTINUOUS_HIGH_RES_MODE, 2)
        lux = ((data[0] << 8) + data[1]) / 1.2
        return lux

# --- DHT22 Setup ---
dhtDevice = adafruit_dht.DHT22(board.D27)  # GPIO27 (Pin 13)

# --- Main Loop ---
try:
    while True:
        try:
            # Read all sensors
            water_temp = read_ds18b20_temp()
            light = read_light()
            air_temp = dhtDevice.temperature
            humidity = dhtDevice.humidity

            # Display all readings
            print(f"💧 Water Temp: {water_temp:.2f}°C")
            print(f"💡 Light Intensity: {light:.2f} lx")
            print(f"🌡️ Air Temp: {air_temp:.2f}°C | 💦 Humidity: {humidity:.2f}%")
            print("-" * 50)

        except RuntimeError as error:
            print(f"DHT22 error: {error.args[0]}")
        
        time.sleep(2)

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    dhtDevice.exit()
