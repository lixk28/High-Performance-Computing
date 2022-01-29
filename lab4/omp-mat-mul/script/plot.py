import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  size_list = [512, 1024, 1536, 2048]
  thread_list = [1, 2, 3, 4, 5, 6, 7, 8]

  for thread in thread_list:
    file_name = "./asset/time_" + str(thread)
    data = open(file_name, 'r')

    gemm_time = []
    ompmm_time = []

    for line in data:
      line = line.strip().split()
      gemm_time.append(float(line[0]))
      ompmm_time.append(float(line[1]))
    
    plt.plot(size_list, gemm_time, marker='x', color='red', label='gemm')
    plt.plot(size_list, ompmm_time, marker='^', color='blue', label='ompmm')

    plt.grid(linestyle='--')       
    plt.legend()                    
    plt.xlabel("Matrix Size")       
    plt.ylabel("Elapsed Time / s")       
    plt.title("Performance (#thread={})".format(thread))    
    plt.savefig("./asset/Performance_{}.svg".format(thread))         
    plt.close()
    
    data.close()