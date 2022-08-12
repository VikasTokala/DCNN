import time
import sys
import os 
import tempfile as tmp

home = os.path.expanduser('~')

if __name__ == '__main__':
    
    sig_dir = home + '/signals/NUANCE_SIGNALS/all'
    work_dir = home + '/mac/hpc/rtf_estimators/fblms/'
    SNR_db_list = [12, 6, 0, -6]

    N_jobs = 50 # 1 job does a list of files for all SNR and noise types
    noise_type = 'white'

    sigs_list = []

    for file in os.listdir(sig_dir):
        # check only text files
        if file.endswith('.wav'):
            sigs_list.append(sig_dir + '/' + file)

    N_sigs = len(sigs_list)

    batch_size = int(N_sigs/N_jobs)

    # Loop over your jobs
    for i in range(N_jobs):

        sigs_batch = sigs_list[i*batch_size:(i+1)*batch_size]

        # Customize your options here
        job_name = "rtf_est_%d" % i
        walltime = '12:00:00'
        processors = "-lselect=1:ncpus=8:mem=32gb"
        command = "python3 /rds/general/user/dtj20/home/mac/hpc/rtf_estimators/fblms/fblms_exp.py " + str(batch_size) + ' '
        for b in range(batch_size):
            command += str(sigs_batch[b]) + ' '    
        for s in range(len(SNR_db_list)):
            command += str(SNR_db_list[s]) + ' '
        command += work_dir

        job_string = """#!/bin/bash
#PBS -N %s
#PBS -lwalltime=%s
#PBS %s
cd $PBS_O_WORKDIR
module load anaconda3/personal
source activate ml
%s""" % (job_name, walltime, processors, command)

        with tmp.NamedTemporaryFile(suffix='.pbs', prefix = work_dir + str(i) + '_') as f:
            f.write(bytes(job_string, 'utf-8'))
            f.flush()
            os.system('qsub ' + f.name)