import subprocess
import time

def get_gpu_memory_usage():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
        return int(output.decode('utf-8').strip())
    except:
        return None

def monitor_gpu_usage(duration=60, interval=1):
    max_gpu_usage = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        usage = get_gpu_memory_usage()
        if usage is not None:
            max_gpu_usage = max(max_gpu_usage, usage)
            print(f"Current GPU memory usage: {usage} MB")
        else:
            print("Failed to get GPU memory usage")
        time.sleep(interval)

    return max_gpu_usage
