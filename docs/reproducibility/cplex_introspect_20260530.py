from docplex.mp.model import Model
import os
m = Model()
print("parallel (0=auto/det, 1=det, -1=opportunistic):", m.parameters.parallel.get())
print("threads (0=auto=all cores):", m.parameters.threads.get())
print("cpu_count:", os.cpu_count())
