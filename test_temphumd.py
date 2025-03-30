import time
import board
import adafruit_dht

# Using GPIO27
dht_device = adafruit_dht.DHT22(board.D27)

while True:
    try:
        temperature = dht_device.temperature
        humidity = dht_device.humidity
        print(f"Temp: {temperature:.1f}°C, Humidity: {humidity:.1f}%")
    except RuntimeError as e:
        print(f"Read error: {e}")
    time.sleep(2)
