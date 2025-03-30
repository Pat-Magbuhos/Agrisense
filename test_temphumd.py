import time
import adafruit_dht
import board

# GPIO22 (Pin 15)
dhtDevice = adafruit_dht.DHT22(board.D27)

try:
    while True:
        try:
            temperature_c = dhtDevice.temperature
            humidity = dhtDevice.humidity
            print(f"Temp: {temperature_c:.2f} °C | 💧 Humidity: {humidity:.2f}%")
        except RuntimeError as error:
            print(f"Sensor error: {error.args[0]}")
            time.sleep(2)
            continue

        time.sleep(2)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    dhtDevice.exit()
