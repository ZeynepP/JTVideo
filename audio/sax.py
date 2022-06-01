import json
import multiprocessing
import time

import datasets
import torch
import torchaudio
from datasets import load_dataset
from saxpy.sax import is_mindist_zero, idx2letter
from collections import defaultdict
from saxpy.strfunc import idx2letter
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.alphabet import cuts_for_asize
import numpy as np
from saxpy.sax import ts_to_string

import  os

from tqdm import tqdm
WORKING_DIR="/usr/src/temp/"
#WORKING_DIR = "./"
DATA_FOLDER = "/usr/src/temp/data/"
#DATA_FOLDER = "./data/"
TRAIN = os.path.join(DATA_FOLDER ,"train.json.csv")
VALIDATION = os.path.join(DATA_FOLDER, "val.json.csv")
INPUT_COL = "audio"
LABEL_COL = "label"
WIN_SIZE = 48000
WIN_SLIDE_SIZE = WIN_SIZE
PAA_LIST =[4,8,16,32]
ALPHABET_SIZE = 16

def sax_via_window(series, win_size, win_slide_size, paa_list, alphabet_size=3,
                   nr_strategy='exact', znorm_threshold=0.01, sax_type='unidim'):

    # Convert to numpy array.
    series = np.array(series)

    # Check on dimensions.
    if len(series.shape) > 2:
        raise ValueError('Please reshape time-series to stack dimensions along the 2nd dimension, so that the array shape is a 2-tuple.')
    # Breakpoints.
    cuts = cuts_for_asize(alphabet_size)

    # Dictionary mapping SAX words to indices.
    sax = defaultdict(list)


    # Sliding window across time dimension.
    prev_word = ''
    nwindows = int(len(series)/win_slide_size)

    for i in range(nwindows):

        # Subsection starting at this index.
        #print(i * win_size, (i + 1) * win_size)
        sub_section = series[i * win_size: (i + 1) * win_size]
        # plt.figure()
        # plt.plot(sub_section.ravel(), "b-")
        # plt.title("Raw time series")
        # plt.show()

        # Z-normalized subsection.
        zn = znorm(sub_section, znorm_threshold)

        # PAA representation of subsection.
        for paa_size in paa_list:
          paa_rep = paa(zn, paa_size)

        # SAX representation of subsection.
          curr_word = ts_to_string(paa_rep, cuts)
      #  print(curr_word)

          sax[str(paa_size)].append(curr_word)


    for s in sax.keys():
        concat  =  " ".join(sax[s])
        sax[s] = concat
    sax["nwindows"] = nwindows
    return sax

import os
def mix_down_if_necessary( signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def label_to_id(label, label_list):
    "map label to id int"

    return label_list.index(label)

def speech_file_to_array(path):

        "resample audio to match what the model expects (16000 khz)"
        signal, sampling_rate = torchaudio.load(path)
        #print("sampling_rate",sampling_rate)
        signal = mix_down_if_necessary(signal)
        sax ={}
        sax["1"] = sax_via_window(signal[0], WIN_SIZE , WIN_SLIDE_SIZE, PAA_LIST, ALPHABET_SIZE)
    #    sax["5"] = sax_via_window(signal[0], WIN_SIZE * 5, WIN_SLIDE_SIZE * 5, PAA_LIST, ALPHABET_SIZE)
        sax["10"] = sax_via_window(signal[0], WIN_SIZE* 10, WIN_SLIDE_SIZE * 10, PAA_LIST, ALPHABET_SIZE)
     #   sax["20"] = sax_via_window(signal[0], WIN_SIZE * 20, WIN_SLIDE_SIZE * 20, PAA_LIST, ALPHABET_SIZE)
        return sax


from itertools import zip_longest
def process_chunk(chunk):
    print(DATA_FOLDER + str(time.time()) +"_train.json")
    with open(DATA_FOLDER + str(time.time() )+"_train.json", mode="a+") as fout:

        for line in chunk:
            try:
                audio_path, label = line.split(",")

                if os.path.exists(audio_path):
                    try:
                        signal = speech_file_to_array(audio_path)
                        signal["id"] = audio_path
                        signal["label"] = label.replace("\n", "")
                        json.dump(signal, fout)
                        fout.write('\n')
                    except Exception as ex:
                        print(ex)
                        print("[EX]", audio_path)
                else:
                    print(audio_path)
            except:
                print(line)


def preprocess(path, output_path):
    "preprocess hf dataset/load data"
    p = multiprocessing.Pool(12)
    with open(path, encoding='utf-8') as file:

        p.map(process_chunk, zip_longest(*[file] * 1000) )










if __name__ == "__main__":
    ################### LOAD DATASETS
    #######
    ####

   # output = DATA_FOLDER + "48000_val.json"
   # preprocess(VALIDATION, output)

    output = DATA_FOLDER + "48000_train.json"
    preprocess(TRAIN, output)

