import time
from jtop import jtop

def log_jetson_stats(interval=2):
    try:
        with jtop(interval=interval) as jetson:
            while jetson.ok():
                stats = []
                for process in jetson.processes:
                    if process[9] == 'python3':
                        current_time = int(time.time())
                        pid = process[0]
                        cpu_usage = process[6]
                        total_memory_usage_mb = process[7] / 1024
                        gpu_memory_usage_mb = process[8] / 1024

                        stats.append(f"{current_time}|{pid}|{cpu_usage:.2f}|{total_memory_usage_mb:.2f}|{gpu_memory_usage_mb:.2f}")

                # Get GPU status
                if jetson.gpu:
                    gpu = list(jetson.gpu.values())[0]
                    load = gpu['status']['load']
                    stats.append(f"GPU Load: {load:.1f}%")

                # Get RAM details
                parameter = jetson.memory['RAM']
                max_val = parameter.get("tot", 100) / 1024  # Convert to MB
                use_val = parameter.get("used", 0) / 1024  # Convert to MB
                cpu_val = (parameter.get("used", 0) - parameter.get("shared", 0)) / 1024  # Convert to MB

                stats.append(f"RAM: GPU={use_val:.2f}MB, CPU={cpu_val:.2f}MB , TOTAL={max_val:.2f}MB")

                # Return collected stats as a string with each entry separated by newline
                result = "\n".join(stats)
                print(result, flush=True)
                time.sleep(interval)
    except Exception as e:
        print(f"Error: {str(e)}", flush=True)

if __name__ == '__main__':
    log_jetson_stats()
