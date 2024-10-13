import sys
import shlex
import subprocess
from knockknock import discord_sender

# notice discord
webhook_url = "https://discord.com/api/webhooks/1009749385170665533/m4nldXOXR5f9iWaXoCDLNGhNI48XEpy-Y9CcBpdFJW_xUipS54LCzXX9xZaCY6IH0vSl"
@discord_sender(webhook_url=webhook_url)
def finish(message):
    return message  # Optional return value

def split(s):
    params = shlex.split(s)
    print(params)
    return params


# cmd_lst = ['--data ./TDC/DTA/DAVIS/random --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name DAVIS --task_name DTA --project_name DAVIS_random_DSN --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --architecture DSN',
#            '--data ./TDC/DTA/DAVIS/cold_split/Drug --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name DAVIS --task_name DTA --project_name DAVIS_cold_drug_DSN --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --architecture DSN',
#            '--data ./TDC/DTA/DAVIS/cold_split/Target --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name DAVIS --task_name DTA --project_name DAVIS_cold_target_DSN --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --architecture DSN',
#            '--data ./TDC/DTA/KIBA/random --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name KIBA --task_name DTA --project_name KIBA_random_DSN --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --architecture DSN',
#            '--data ./TDC/DTA/KIBA/cold_split/Drug --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name KIBA --task_name DTA --project_name KIBA_cold_drug_DSN --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --architecture DSN',
#            '--data ./TDC/DTA/KIBA/cold_split/Target --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name KIBA --task_name DTA --project_name KIBA_cold_target_DSN --n_workers 10 --n_epochs 200 --lr 1e-3 --batch_size 1024 --architecture DSN',
#            ]

cmd_lst = ['--data ./TDC/DTA/DAVIS/random --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name DAVIS --task_name DTA --project_name DAVIS_random_DSN --n_workers 4 --n_epochs 200 --lr 1e-3 --batch_size 512 --architecture DSN',
           '--data ./TDC/DTA/DAVIS/cold_split/Drug --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name DAVIS --task_name DTA --project_name DAVIS_cold_drug_DSN --n_workers 4 --n_epochs 200 --lr 1e-3 --batch_size 512 --architecture DSN',
           '--data ./TDC/DTA/DAVIS/cold_split/Target --d1_col Drug --d2_col Target_ID --d1_type Drug --d2_type Protein --data_name DAVIS --task_name DTA --project_name DAVIS_cold_target_DSN --n_workers 4 --n_epochs 200 --lr 1e-3 --batch_size 512 --architecture DSN'
           ]

import time
for command in cmd_lst:
    start = time.time()
    finish(f'\nstarting {command}\n')
    subprocess.run(args=[sys.executable, 'run.py'] + split(command))

    f_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    finish(f'\nFinished: {command}\nLearning time: {f_time}')
finish(f'\nAll job finished successfully!')
