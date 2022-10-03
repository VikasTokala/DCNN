import numpy as np
import os
import soundfile as sf
import scipy.signal as sig
import tempfile as tmp

fs = 16000

ace_folder = '/rds/general/user/dtj20/home/signals/ACE/Lin8Ch/'
signal_folder = '/rds/general/user/dtj20/home/signals/LSP/'
output_folder = '/rds/general/user/dtj20/home/mac/datasets/ace_librispeech/'

N_signals = 50

signal_paths = []
for dirpath, dirnames, filenames in os.walk(signal_folder):
    for filename in [f for f in filenames if f.endswith('.wav')][:N_signals]:
        signal_paths.append(signal_folder + filename)

for i, s in enumerate(signal_paths):

    job_name = "ace_libri_%d" % i
    walltime = '02:00:00'
    processors = "-lselect=1:ncpus=16:mem=32gb"
    command = "python3 /rds/general/user/dtj20/home/mac/hpc/ace_librispeech/gen_ace_librispeech.py" + ' ' + s + ' ' + str(fs) + ' ' + output_folder 

    job_string = """#!/bin/bash
    #PBS -N %s
    #PBS -lwalltime=%s
    #PBS %s
    cd $PBS_O_WORKDIR
    module load anaconda3/personal
    source activate ml
    %s""" % (job_name, walltime, processors, command)

    with tmp.NamedTemporaryFile(suffix='.pbs', prefix = output_folder + str(i) + '_') as f:
        f.write(bytes(job_string, 'utf-8'))
        f.flush()
        os.system('qsub ' + f.name)