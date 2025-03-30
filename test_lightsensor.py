import time
import bh1750

sensor = bh1750.BH1750()

try:
    while True:
        lux = sensor.luminance(bh1750.CONT_HIRES_1)
        print(f"Light Intensity: {lux:.2f} lux")
        time.sleep(1)

except KeyboardInterrupt:
    print("Measurement stopped by user")
