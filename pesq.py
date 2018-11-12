import os
import sys
import subprocess
import re
from glob import glob
import numpy as np


def pesq(source_file_path, estimation_file_path, pesq_bin_path, mode='wb'):
    if mode == 'nb':
        command_args = [pesq_bin_path, "+16000", source_file_path, estimation_file_path]
    else:
        command_args = [pesq_bin_path, "+16000", "+wb", source_file_path, estimation_file_path]
    
    try:
        output = subprocess.check_output(command_args)

        if mode == 'nb':
            match = re.search("\(Raw MOS, MOS-LQO\):\s+= (-?[0-9.]+?)\t([0-9.]+?)$", output.decode().replace('\r', ''), re.MULTILINE)
            mos = float(match.group(1))
            moslqo = float(match.group(2))
            return mos, moslqo
        else:
            match = re.search("\(MOS-LQO\):\s+= ([0-9.]+?)$", output.decode().replace('\r', ''), re.MULTILINE)
            mos = float(match.group(1))
            return mos, None
    except subprocess.CalledProcessError:
        return None, None


def eval_pesq(audio_folder, pesq_bin_path, mode='wb'):
    target_folder = os.path.join(audio_folder, 'target')
    mixed_folder = os.path.join(audio_folder, 'mixed')
    masked_folder = os.path.join(audio_folder, 'masked')
    
    filenames = glob(os.path.join(target_folder, '*.wav'))
    mixed_mos = []
    masked_mos = []
    
    for target_fname in filenames:
        mixed_fname = target_fname.replace(target_folder, mixed_folder)
        masked_fname = target_fname.replace(target_folder, masked_folder)
        m_mixed, _ = pesq(target_fname, mixed_fname, pesq_bin_path, mode)
        m_masked, _ = pesq(target_fname, masked_fname, pesq_bin_path, mode)
        
        if m_mixed is not None and m_masked is not None:
            print("Noisy PESQ: %f - enhancement PESQ: %f" % (m_mixed, m_masked))
            mixed_mos.append(m_mixed)
            masked_mos.append(m_masked)
        
    print("#########################################")
    print("Number of samples: %d" % len(masked_mos))
    print("Enhancement mean (std) PESQ: %f (%f)" % (np.mean(masked_mos), np.std(masked_mos)))
    print("Noisy mean (std) PESQ: %f (%f)" % (np.mean(mixed_mos), np.std(mixed_mos)))
    print("#########################################")
    
    return mixed_mos, masked_mos, filenames