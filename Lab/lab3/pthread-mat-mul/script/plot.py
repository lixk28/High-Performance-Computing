import matplotlib.pyplot as plt

if __name__ == "__main__":
  scale_list = [512 * i for i in range(1, 5)]
  thread_list = [i for i in range(1, 9)]

  for thread in thread_list:
    serial_time_list = []
    parallel_time_list = []

    file_name = "./asset/time-" + str(thread)
    file = open(file_name, 'r')
    serial_time_list = []
    parallel_time_list = []

    for line in file:
      data = line.strip().split()
      serial_time_list.append(float(data[0]))
      parallel_time_list.append(float(data[1]))

    file.close()

    plt.plot(scale_list, serial_time_list, marker='x', color='red', label='serial')
    plt.plot(scale_list, parallel_time_list, marker='^', color='green', label='pthread')
    plt.grid(linestyle='--')       
    plt.legend()                
    plt.xlabel("Matrix Scale")       
    plt.ylabel("Wall Time / s")        
    plt.title("Performance" + "(thread={})".format(thread))     
    plt.savefig("./asset/performance-{}.svg".format(thread))        
    plt.show()                    
