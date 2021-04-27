import multiprocessing as mp
import psutil
import time

def my_func(x):
  print(mp.current_process())
  time.sleep(x)
  return x**x

def main():
    n_cpus = psutil.cpu_count(logical=True)
    print(f"We have {n_cpus} cpus")
    pool = mp.Pool(n_cpus)
    result = pool.map(my_func, [4,2,3,5,3,2,1,2])
    print(result)
    result_set_2 = pool.map(my_func, [4,6,5,4,6,3,23,4,6])
    print(result_set_2)
    result_set_3 = pool.map(my_func, [4, 6, 5, 4, 6, 3, 2])
    print(result_set_3)


if __name__ == "__main__":
  main()
