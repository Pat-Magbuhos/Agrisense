import time
import glob
import board
import adafruit_dht
from smbus2 import SMBus
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db

# --- Load environment variables from .env ---
dotenv_path = os.path.join(os.path.dirname(__file__), "venv/.env")
load_dotenv(dotenv_path)

# --- Retrieve Firebase credentials from .env ---
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
SERVICE_ACCOUNT_PATH = os.getenv("SERVICE_ACCOUNT_PATH", "venv/serviceAccountKey.json")

# --- Validate environment variables ---
if not FIREBASE_DB_URL:
    raise ValueError("ERROR: FIREBASE_DB_URL is missing from .env!")
if not os.path.exists(SERVICE_ACCOUNT_PATH):
    raise ValueError(f"ERROR: Service account key not found at {SERVICE_ACCOUNT_PATH}")

# --- Initialize Firebase ---
try:
    firebase_admin.delete_app(firebase_admin.get_app())
except ValueError:
    pass  # No app was initialized

cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})

# --- DS18B20 Setup ---
base_dir = '/sys/bus/w1/devices/'
device_folders = glob.glob(base_dir + '28*')
if not device_folders:
    raise RuntimeError("DS18B20 sensor not found. Check wiring and 1-Wire setup.")

device_folder = device_folders[0]
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

def upload_sensor_data_to_firebase(timestamp, water_temp, light, air_temp, humidity):
    firebase_path = f"detections/{timestamp}/environment_data"
    ref = db.reference(firebase_path)
    ref.set({
        "water_temperature_c": round(water_temp, 2),
        "light_intensity_lux": round(light, 2),
        "air_temperature_c": round(air_temp, 2),
        "humidity_percent": round(humidity, 2)
    })
    print(f"Uploaded environment data to Firebase under {firebase_path}")

# --- Continuous Logging Every 10 Minutes ---
if __name__ == "__main__":
    try:
        while True:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            water_temp = read_ds18b20_temp()
            light = read_light()
            air_temp = dhtDevice.temperature
            humidity = dhtDevice.humidity
            upload_sensor_data_to_firebase(timestamp, water_temp, light, air_temp, humidity)
            time.sleep(600)  # 10 minutes = 600 seconds
    except KeyboardInterrupt:
        print("Sensor logging stopped by user.")
    finally:
        dhtDevice.exit()
