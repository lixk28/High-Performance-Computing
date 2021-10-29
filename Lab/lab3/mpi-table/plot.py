import matplotlib.pyplot as plt

if __name__ == "__main__":
  proc_list = [1, 2, 4, 8, 16]
  order_list = [128, 256, 512, 1024, 2048]

  gemm_time = [0.020905, 0.138029, 1.132524, 28.505833, 403.162180]
  optmm_time = [0.011226, 0.039191, 0.198281, 2.693320, 43.000741]

  mpi_p2p_time = {
    1: [0.013805, 0.132869, 1.136721, 27.440135, 405.727127],
    2: [0.008255, 0.067919, 0.643046, 17.992966, 216.574897],
    4: [0.006454, 0.051333, 0.521087, 11.130705, 126.646435],
    8: [0.005213, 0.037429, 0.453798, 8.402441, 91.816284],
    16: [0.124014, 0.208111, 0.888161, 8.640943, 89.016351]
  }
  mpi_collect_time = {
    1: [0.014632, 0.137869, 1.148502, 28.705399, 411.752261],
    2: [0.016707, 0.073242, 0.610015, 15.051788, 213.683344],
    4: [0.007334, 0.064321, 0.553858, 9.156726, 125.055034],
    8: [0.011986, 0.036423, 0.465381, 6.575210, 92.940979],
    16: [0.67960, 0.260153, 0.908136, 7.515069, 110.062592]
  }

  optmm_speedups = []
  mpi_p2p_speedups = []
  mpi_collect_speedups = []

  for i in range(5):
    optmm_speedups.append(gemm_time[i] / optmm_time[i])

  for i in range(5):
    mpi_p2p_speedups.append(gemm_time[i] / mpi_p2p_time[4][i])
    mpi_collect_speedups.append(gemm_time[i] / mpi_collect_time[4][i])

  plt.plot(order_list, optmm_speedups, marker='.', color='red', label='optmm')
  plt.plot(order_list, mpi_p2p_speedups, marker='o', color='green', label='mpi_p2p')
  plt.plot(order_list, mpi_collect_speedups, marker='x', color='blue', label='mpi_collect')

  plt.grid(linestyle='--')       
  plt.legend()                   
  plt.xlabel("Matrix Order (#processes = 4)")       
  plt.ylabel("Speedup")        
  plt.title("Parallel Speedups")    
  plt.savefig("speedup.svg")    
  plt.close()

  mpi_p2p_efficiencies = []
  mpi_collect_efficiencies = []

  for speedup in mpi_p2p_speedups:
    mpi_p2p_efficiencies.append(speedup / 4)
  for speedup in mpi_collect_speedups:
    mpi_collect_efficiencies.append(speedup / 4)

  plt.plot(order_list, mpi_p2p_efficiencies, marker='s', color='green', label='mpi_p2p')
  plt.plot(order_list, mpi_collect_efficiencies, marker='^', color='blue', label='mpi_collect')

  plt.grid(linestyle='--')       
  plt.legend()                   
  plt.xlabel("Matrix Order (#processes = 4)")       
  plt.ylabel("Efficiency")        
  plt.title("Parallel Effciencies")    
  plt.savefig("efficiency.svg")
  plt.close()
