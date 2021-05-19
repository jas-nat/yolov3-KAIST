import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

c_time=[8.42,9.99,11.87, 11.42, 12.85]
precision = [64.9,70.7, 71.5, 72.1, 72.5]

plt.plot(c_time[4], precision[4], marker="o", color="b", label ="YOLO - 4L", markersize=10) #circle
plt.plot(c_time[3], precision[3], marker="v", color="m", label ="YOLO - 4L - C1", markersize = 10) #Diamond
plt.plot(c_time[2], precision[2], marker="D", color="g", label ="YOLO - 4L - C2", markersize = 10) #hexagon
plt.plot(c_time[1], precision[1], marker="h", color="k", label ="YOLO - 4L - C3", markersize = 10) #triangle

plt.plot(c_time[0], precision[0], marker="s", color="y", label ="YOLO - 4L - C4", markersize = 10)#square

# plt.ylim(top=80, bottom=60, auto=True)
# plt.xlim(left=0.008, right=0.015, auto=True)
plt.legend(loc='lower right')
plt.xlabel("processing time (ms)")
plt.ylabel("accuracy (%)")
plt.show()