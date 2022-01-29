import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  elapsed_time_general_avg_list = []
  elapsed_time_strassen_avg_list = []
  elapsed_time_opt_avg_list = []
  elapsed_time_mkl_avg_list = []
  scale_list = []

  asset_path = "../asset/"
  file_name_list = os.listdir(asset_path)
  for file in file_name_list:
    # print(file)
    scale = int(file.replace("elapsed_time_", ''))
    scale_list.append(scale)
    data = open(asset_path + file, 'r')
    elapsed_time_general_total = 0.0
    elapsed_time_strassen_total = 0.0
    elapsed_time_opt_total = 0.0
    elapsed_time_mkl_total = 0.0
    # read elapsed time from data
    for line in data:
      elapsed_time_line = line.strip().split()
      # for t in elapsed_time_line:
      #   print(t)
      elapsed_time_general_total += float(elapsed_time_line[0])
      elapsed_time_strassen_total += float(elapsed_time_line[1])
      elapsed_time_opt_total += float(elapsed_time_line[2])
      elapsed_time_mkl_total += float(elapsed_time_line[3])
    # compute average time of 5 test case
    elapsed_time_general_avg_list.append(elapsed_time_general_total / 5)
    elapsed_time_strassen_avg_list.append(elapsed_time_strassen_total / 5)
    elapsed_time_opt_avg_list.append(elapsed_time_opt_total / 5)
    elapsed_time_mkl_avg_list.append(elapsed_time_mkl_total / 5)
  
  plt.xlabel("Matrix Scale")
  plt.ylabel("Elapsed Time / s")
  plt.scatter(scale_list, elapsed_time_general_avg_list, color='red', marker='x')
  plt.scatter(scale_list, elapsed_time_strassen_avg_list, color='orange', marker='s')
  plt.scatter(scale_list, elapsed_time_opt_avg_list, color='blue', marker='^')
  plt.scatter(scale_list, elapsed_time_mkl_avg_list, color='green', marker='o')
  plt.show()
