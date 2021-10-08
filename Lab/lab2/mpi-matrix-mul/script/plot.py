import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  scale_list = [256 * i for i in range(4, 9)]
  proc_list = [1, 2, 4]
  file_path = "./asset/"

  for proc in proc_list:
    distb_time_p2p_list = []
    merge_time_p2p_list = []
    comm_time_p2p_list = []
    elapsed_time_p2p_list = []
    serial_time_p2p_list = []

    distb_time_clc_list = []
    merge_time_clc_list = []
    comm_time_clc_list = []
    elapsed_time_clc_list = []
    serial_time_clc_list = []

    for scale in scale_list:
      file_name_p2p = "p2p_" + str(scale) + '_' + str(proc)
      file_name_clc = "collect_" + str(scale) + '_' + str(proc)
      data_p2p = open(file_path + file_name_p2p, 'r')
      data_clc = open(file_path + file_name_clc, 'r')

      distb_time_p2p = 0.0
      merge_time_p2p = 0.0
      comm_time_p2p = 0.0
      elapsed_time_p2p = 0.0
      serial_time_p2p = 0.0
      test_case = 0
      for line in data_p2p:
        test_case += 1
        time = line.strip().split()
        distb_time_p2p += float(time[0])
        merge_time_p2p += float(time[1])
        comm_time_p2p += float(time[2])
        elapsed_time_p2p += float(time[3])
        serial_time_p2p += float(time[4])
      distb_time_p2p_list.append(distb_time_p2p / test_case)
      merge_time_p2p_list.append(merge_time_p2p / test_case)
      comm_time_p2p_list.append(comm_time_p2p / test_case)
      elapsed_time_p2p_list.append(elapsed_time_p2p / test_case)
      serial_time_p2p_list.append(serial_time_p2p / test_case)
      
      distb_time_clc = 0.0
      merge_time_clc = 0.0
      comm_time_clc = 0.0
      elapsed_time_clc = 0.0
      serial_time_clc = 0.0
      test_case = 0
      for line in data_clc:
        test_case += 1
        time = line.strip().split()
        distb_time_clc += float(time[0])
        merge_time_clc += float(time[1])
        comm_time_clc += float(time[2])
        elapsed_time_clc += float(time[3])
        serial_time_clc += float(time[4])
      distb_time_clc_list.append(distb_time_clc / test_case)
      merge_time_clc_list.append(merge_time_clc / test_case)
      comm_time_clc_list.append(comm_time_clc / test_case)
      elapsed_time_clc_list.append(elapsed_time_clc / test_case)
      serial_time_clc_list.append(serial_time_clc / test_case)
      
      data_p2p.close()
      data_clc.close()
  
    plt.xlabel("Matrix Scale" + " (#process = {})".format(proc))
    plt.ylabel("Time/s")
    plt.scatter(scale_list, distb_time_p2p_list, color='red', marker='x')
    plt.scatter(scale_list, merge_time_p2p_list, color='orange', marker='s')
    plt.scatter(scale_list, comm_time_p2p_list, color='yellow', marker='^')
    plt.scatter(scale_list, elapsed_time_p2p_list, color='green', marker='o')
    plt.scatter(scale_list, serial_time_p2p_list, color='blue', marker='v')
    plt.savefig("./asset/p2p_proc" + str(proc) + '.png')
    plt.show()

    plt.xlabel("Matrix Scale" + " (#process = {})".format(proc))
    plt.ylabel("Time/s")
    plt.scatter(scale_list, distb_time_clc_list, color='red', marker='x')
    plt.scatter(scale_list, merge_time_clc_list, color='orange', marker='s')
    plt.scatter(scale_list, comm_time_clc_list, color='yellow', marker='^')
    plt.scatter(scale_list, elapsed_time_clc_list, color='green', marker='o')
    plt.scatter(scale_list, serial_time_clc_list, color='blue', marker='v')
    plt.savefig("./asset/clc_proc" + str(proc) + '.png')
    plt.show()
