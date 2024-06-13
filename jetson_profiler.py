import sys
from jtop import jtop
from time import sleep

def get_jetson_stats(pid):
    with jtop(interval=0.05) as jetson:
        while True:
            for p in jetson.processes:
                if p[0] == pid:
                    print(f"{p[6]}|{p[7]}|{p[8]}|{list(jetson.gpu.values())[0]['status']['load']}", flush=True)
            sleep(0.05)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python jetson_profiler.py <pid>")
        sys.exit(1)
    pid = int(sys.argv[1])
    get_jetson_stats(pid)
