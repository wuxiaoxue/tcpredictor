
import csv
import pandas
import numpy as np
from sklearn.utils import shuffle
from Tools.scripts.objgraph import ignore
from scipy import sparse


data_name = "hdfs"


input_file0 = "../datasource/hdfs_0.csv"
input_file1 = "../datasource/hdfs_1.csv"

col_names = ['content','label']
sourcedata0 = pandas.read_csv(input_file0, names=col_names).fillna('')
#sourcedata0 = sourcedata0[:4000]  # for large sized imbalance data
sourcedata1 = pandas.read_csv(input_file1, names=col_names).fillna('')
#sourcedata2 = shuffle(sourcedata2)

data = np.vstack((sourcedata0, sourcedata1))
data = shuffle(data)
data_content = data[:,0]
data_label = data[:,1]

out = open("../input/" + data_name + "_labeled.csv", "w", newline='')
writer = csv.writer(out, delimiter=',')
for item in range(0, len(data_label)-1):
    row = [data_content[item], data_label[item]]
#    if not(regexp.search(sourcedata_content[item])):
    writer.writerow(row)
out.close()

print("wonderful")

