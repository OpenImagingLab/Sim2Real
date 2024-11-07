import os
import sys
import datetime
import random
from multiprocessing import Process, Queue, current_process
from natsort import natsorted as sorted

import sys
sys.path.append(os.getcwd())
# Assuming your imports are handled correctly
from config.GoPRO.GOPRO_config import source_path
from config.GoPRO.GOPRO_SyntheticInput_train import GOPRO_SyntheticInput_train

# Function to perform the task
def process_task(folder_name, formatted_time):
    print(f"-------- Processing by {current_process().name}: {folder_name}")
    try:
        # Fixing parameters and generating randomly is undesirable.
        leak_rate = 0.1 #random.uniform(0.005, 0.02)
        shot_noise_rate = 10 #random.uniform(0.05, 0.2)
        cutoff = 1000 #random.uniform(100, 5000)
        sigma_thr = 0.03 #random.uniform(0.04, 0.08)
        pos_thr = 0.25 #random.uniform(0.2, 0.3)
        neg_thr = 0.25 #random.uniform(0.2, 0.3)
        
        os.system(f'python v2e_simpler.py '
                  f'--leak_rate_hz={leak_rate} '
                  f'--shot_noise_rate_hz={shot_noise_rate} '
                  f'--cutoff_hz={cutoff} '
                  f'--sigma_thr={sigma_thr} '
                  f'--pos_thr={pos_thr} '
                  f'--neg_thr={neg_thr} '
                  f'--output_width=1280 '
                  f'--output_height=720 '
                  f'--synthetic_input=config.GoPRO.GOPRO_SyntheticInput_train '
                  f'--file_key={folder_name} '
                  f'--dvs_aedat2=None '
                  f'--no_preview '
                  f'--refractory_period=0.00001')
    except Exception as e:
        print(f"Error processing folder {folder_name}: {e}")

# Function to start worker processes
def start_workers(num_workers, q, formatted_time):
    processes = []
    for _ in range(num_workers):
        p = Process(target=worker, args=(q, formatted_time))
        processes.append(p)
        p.start()
    return processes

# Main worker function
def worker(q, formatted_time):
    while True:
        folder_name = q.get()
        if folder_name is None:
            break
        process_task(folder_name, formatted_time)

# Main function
if __name__ == "__main__":
    # num_workers = 22
    keys = sorted(os.listdir(source_path.train.source_path))
    total = len(keys)
    
    # num_workers = len(keys)
    num_workers = 11

    print('-'*20)
    print(f'Totally we have {total} folders\nwhich are: {keys}')
    print('-'*20)

    q = Queue()
    for k in keys:
        q.put(k)

    # Send a 'None' for each worker to signal them to stop
    for _ in range(num_workers):
        q.put(None)

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d_%H-%M')

    processes = start_workers(num_workers, q, formatted_time)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print('Processing complete.')


