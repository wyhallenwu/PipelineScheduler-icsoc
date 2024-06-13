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

        # while jetson.ok():
        #     stats = []
        #     for process in jetson.processes:
        #         if process[0] == pid:
        #             current_time = int(time.time())
        #             cpu_usage = process[6]
        #             total_memory_usage_mb = process[7] / 1000
        #             gpu_memory_usage_mb = process[8] / 1000
        #
        #             stats.append(
        #                 f"{current_time}|{int(cpu_usage)}|{int(total_memory_usage_mb)}|{int(gpu_memory_usage_mb)}")
        #
        #     # Get GPU status
        #     if jetson.gpu:
        #         gpu = list(jetson.gpu.values())[0]
        #         load = gpu['status']['load']
        #         stats.append(f"GPU Load: {int(load)}%")
        #
        #     # Get RAM details
        #     parameter = jetson.memory['RAM']
        #     max_val = parameter.get("tot", 100) / 1024  # Convert to MB
        #     use_val = parameter.get("used", 0) / 1024  # Convert to MB
        #     cpu_val = (parameter.get("used", 0) - parameter.get("shared", 0)) / 1024  # Convert to MB
        #
        #     stats.append(f"RAM: GPU={int(use_val)}MB, CPU={int(cpu_val)}MB, TOTAL={int(max_val)}MB")
        #
        #     # Return collected stats as a string with each entry separated by newline
        #     result = "\n".join(stats)
        #     print(result, flush=True)
        #     time.sleep(interval)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python jetson_profiler.py <pid>")
        sys.exit(1)
    pid = int(sys.argv[1])
    get_jetson_stats(pid)
