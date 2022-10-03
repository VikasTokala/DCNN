import time
import sys
import os 
import tempfile as tmp

home = os.path.expanduser('~')

if __name__ == '__main__':

    sig_dir = home + '/mac/datasets/ace_librispeech/'
    rir_dirs = ['rir1/', 'rir2/', 'rir3/', 'rir4/']
    SNR_db_list = [12, 6, 0, -6]

    N_jobs_per_rir = 10
    N_signals = 200
    N_rirs = len(rir_dirs)
    batch_size = int(N_signals/(N_jobs_per_rir*N_rirs))

    noise_type = 'ssn'

    sigs_list = []

    # Loop over your jobs
    # for rir_string in rir_dirs:
    for rir_string in [rir_dirs[0]]:

        work_dir = home + '/mac/hpc/rtf_estimators/ssn/pm_gevd/' + rir_string

        for file in os.listdir(sig_dir+rir_string):
                # check only text files
                if file.endswith('.wav'):
                    sigs_list.append(sig_dir+rir_string+file)

        # for i in range(N_jobs_per_rir):
        for i in range(1):

            sigs_batch = sigs_list

            # Customize your options here
            job_name = "rtf_est_ssn_%d" % i
            walltime = '12:00:00'
            processors = "-lselect=1:ncpus=16:mem=32gb"
            command = "python3 /rds/general/user/dtj20/home/mac/hpc/rtf_estimators/ssn/pm_gevd/pm_gevd_exp.py " + str(batch_size) + ' '
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
