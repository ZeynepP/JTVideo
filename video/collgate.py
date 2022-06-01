
import json
import time
import pandas as pd
import requests
import glob
import csv
from datetime import datetime

def collgate_repair(file_input):

#    import IPython;
#    IPython.embed();
#    exit(1)
    with  open(file_input, encoding="utf-8") as flist:
        for line in flist:
            info = line.split("/")
            print(info[1], info[2])

            with open("../text/data/subjects_v3_2017-2019.json", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if str(obj["id"]) in info[2]:
                        code_chaine = obj["media"]
                        data_diff = obj["docTime"]
                        output = "../mp4/" + str(info[1]) + "/" + str(info[2]).strip().replace("'","")
                        print("out", output)

                        try:

                                duree = (pd.Timestamp(obj["endTime"]).tz_convert("UTC") - pd.Timestamp(data_diff).tz_convert(
                                    "UTC")).seconds

                                url = "http://collgate.ina." \
                                      "fr:81/collgate.dlweb/get/" + code_chaine + "/" + data_diff + "/" + str(duree) + "?download"
                                r = requests.get(url)

                                print(url)

                                open(output, "wb").write(r.content)
                                #time.sleep(0.5)
                                break
                        except Exception as ex:
                                print(ex)
                                print("ERROR", obj["id"])



collgate_repair("/usr/src/mp4/0size.txt")
def download_from_scrath():
    with open("../text/data/subjects_v3_2017-2019.json", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            code_chaine = obj["media"]
            data_diff = obj["docTime"]
            files = glob.glob("/usr/src/mp4/")
            output = "/usr/src/mp4/" + str(obj["id"])+'.mp4'
            if output not in files :
                try:

                    duree = (pd.Timestamp(obj["endTime"]).tz_convert("UTC") - pd.Timestamp(data_diff).tz_convert("UTC")).seconds

                    url = "http://collgate.ina." \
                          "fr:81/collgate.dlweb/get/" + code_chaine + "/" + data_diff +"/"  + str(duree) + "?download"
                    r = requests.get(url)

                    print(url)

                    open(output, "wb+").write(r.content)
                    time.sleep(0.5)

                except Exception as ex:
                    print(ex)
                    print("ERROR", obj["id"])
            else:
                    print("Already done ", output)


