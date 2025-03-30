import glob
import time

# DS18B20 file path
base_dir = '/sys/bus/w1/devices/'
device_folder = glob.glob(base_dir + '28*')[0]
device_file = device_folder + '/w1_slave'

def read_temp_raw():
    with open(device_file, 'r') as f:
        lines = f.readlines()
    return lines

def read_temp():
    lines = read_temp_raw()
    while lines[0].strip()[-3:] != 'YES':
        time.sleep(0.2)
        lines = read_temp_raw()
    temp_pos = lines[1].find('t=')
    if temp_pos != -1:
        temp_string = lines[1][temp_pos+2:]
        temp_c = float(temp_string) / 1000.0
        return temp_c

if __name__ == '__main__':
    try:
        while True:
            temp = read_temp()
            print(f"Water Temperature: {temp:.2f}°C")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nMeasurement stopped by user.")
