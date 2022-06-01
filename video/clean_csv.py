import json
import collections

import os
os.environ['http_proxy'] = "http://firewall.ina.fr:81"
os.environ['HTTP_PROXY'] = "http://firewall.ina.fr:81"
os.environ['https_proxy'] = "http://firewall.ina.fr:81"
os.environ['HTTPS_PROXY'] = "http://firewall.ina.fr:81"

#
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification, \
#     TextClassificationPipeline
# from transformers import SummarizationPipeline
# model_name = 'lincoln/flaubert-mlsum-topic-classification'
#
# loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)
# loaded_model = AutoModelForSequenceClassification.from_pretrained(model_name)
#
# nlp = TextClassificationPipeline(model=loaded_model, tokenizer=loaded_tokenizer)
# with open("./test.csv","w+") as fout:
#     with open("./subjects_v3_2017-2019.json","r") as file:
#          for line in file:
#              j = json.loads(line)
#              try:
#                  results = nlp(j["document"])[0]
#                  results["ina"] = j["topics"][0]["label"]
#                  json.dump(results, fout)
#                  fout.write("\n")
#              except:
#                  print(line)

cat = []
with open("../text/data/subjects_v3_2017-2019.json", "r") as file:
    for line in file:
        j = json.loads(line)
        cat.append(len(j["document"].split(" ")))

#
d = collections.Counter(cat)
# print(d)
# less = 0
# great = 0
# for k,v in d.items():
#     if k < 512:
#         less  += v
#     else:
#         great += v
# print(less, great)

import matplotlib.pyplot as plt
import numpy as np
# An "interface" to matplotlib.axes.Axes.hist() method
mc = d.most_common(750)

plt.bar(*zip(*mc))
plt.xlabel("Doc length")
plt.ylabel("Number of counts ")
plt.show()
# d = {'Politique France': 8274, 'Société': 7810, 'International': 7035, 'Catastrophes': 5306, 'Economie': 4078, 'Faits divers': 3223, 'Sport': 2461, 'Justice': 2439, 'Environnement': 1584, 'Santé': 1582, 'Histoire-hommages': 1484, 'Culture-loisirs': 1136, 'Education': 1022, 'Sciences et techniques': 433}
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# #data prep
# plt.interactive(False)
# plt.rcParams.update({'font.size': 14})
# fig, ax = plt.subplots(figsize=(8,6))
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(True)
# ax.spines['left'].set_visible(False)
# colors=mcolors.TABLEAU_COLORS
# ax.barh(list(d.keys()), width=list(d.values()), color=colors)
# plt.title("Distribution of categories");
# plt.show()
# #######################################################################
#

# # to remove not find videos from dataset
# path = "/usr/src/temp/0/"
# import os.path as osp
# for t in ["train","val", "test"]:
#     newout =  open(path + t + "_clean.csv", "w+")
#     with open(path + t + ".csv", "r") as file:
#         for line in file:
#             sline = line.split("\t")
#             print(sline)
#             if not osp.exists(sline[0]):
#                 print(sline[0], " not exists")
#             else:
#                 newout.write(line)
#
#     newout.close()
