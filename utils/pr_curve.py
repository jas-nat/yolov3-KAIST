#plotting
#change label in figure 2

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
conf_day = [] #for rgb
precision_day = []
recall_day = []

conf_night = [] # for ir
precision_night = []
recall_night = []

conf_multi = []
precision_multi = []
recall_multi = []

conf_multi2 = []
precision_multi2 = []
recall_multi2 = []

conf_multi3 = []
precision_multi3 = []
recall_multi3 = []


# file = open("compress/A_2/pr_curve_class0.0.txt", 'r')
file2 = open("night_rgb/visualization/hsv off/pr_curve_class0.0.txt", 'r')
# file3 = open("compress/C/pr_curve_class0.0.txt", 'r')
file4 = open("night_ir/visualization/hsv off 1 channel/pr_curve_class0.0.txt", 'r')
file5 = open("night_multi/visualization/yolov3-spp/Adam/pr_curve_class0.0.txt", 'r')


# contents = file.readlines()
contents2 = file2.readlines()
# contents3 = file3.readlines()
contents4 = file4.readlines()
contents5 = file5.readlines()

# for i, line in enumerate(contents,1):
# 	data = line.split(" ")
# 	conf_day.append(float(data[0]))
# 	precision_day.append( float(data[1]))
# 	recall_day.append(float(data[2]))

for i, line in enumerate(contents2,1):
	data = line.split(" ")
	conf_night.append(float(data[0]))
	precision_night.append( float(data[1]))
	recall_night.append(float(data[2]))

# for i, line in enumerate(contents3,1):
# 	data = line.split(" ")
# 	conf_multi.append(float(data[0]))
# 	precision_multi.append( float(data[1]))
# 	recall_multi.append(float(data[2]))

for i, line in enumerate(contents4,1):
	data = line.split(" ")
	conf_multi2.append(float(data[0]))
	precision_multi2.append( float(data[1]))
	recall_multi2.append(float(data[2]))

for i, line in enumerate(contents5,1):
	data = line.split(" ")
	conf_multi3.append(float(data[0]))
	precision_multi3.append( float(data[1]))
	recall_multi3.append(float(data[2]))



#Find the cross point
best_conf_day = 0
best_precision_day = 0 
best_recall_day = 0

best_conf_night = 0
best_precision_night = 0
best_recall_night = 0

best_conf_multi = 0
best_precision_multi = 0
best_recall_multi = 0

best_conf_multi2 = 0
best_precision_multi2 = 0
best_recall_multi2 = 0

best_conf_multi3 = 0
best_precision_multi3 = 0
best_recall_multi3 = 0

# for i in range(len(conf_day)-1):
# 	if(recall_day[i] == precision_day[i]):
# 		best_conf_day = conf_day[i]
# 		best_precision_day = precision_day[i]
# 		best_recall_day = recall_day[i]
# 		print(f"1 index {i} Conf {conf_day[i]:.3f} Precision {precision_day[i]:.3f} recall {recall_day[i]:.3f}")
for i in range(len(conf_night)-1):
	if(recall_night[i] == precision_night[i]):
		best_conf_night = conf_night[i]
		best_precision_night= precision_night[i]
		best_recall_night = recall_night[i]
		print(f"2 index {i} Conf {conf_night[i]:.3f} Precision {precision_night[i]:.3f} recall {recall_night[i]:.3f}")
# for i in range(len(conf_multi)-1):
# 	if(recall_multi[i] == precision_multi[i]):
# 		best_conf_multi = conf_multi[i]
# 		best_precision_multi = precision_multi[i]
# 		best_recall_multi = recall_multi[i]
# 		print(f"3 index {i} Conf {conf_multi[i]:.3f} Precision {precision_multi[i]:.3f} recall {recall_multi[i]:.3f}")
for i in range(len(conf_multi2)-1):
	if(recall_multi2[i] == precision_multi2[i]):
		best_conf_multi2 = conf_multi2[i]
		best_precision_multi2 = precision_multi2[i]
		best_recall_multi2 = recall_multi2[i]
		print(f"4 index {i} Conf {conf_multi2[i]:.3f} Precision {precision_multi2[i]:.3f} recall {recall_multi2[i]:.3f}")
for i in range(len(conf_multi3)-1):
	if(recall_multi3[i] == precision_multi3[i]):
		best_conf_multi3 = conf_multi3[i]
		best_precision_multi3 = precision_multi3[i]
		best_recall_multi3 = recall_multi3[i]
		print(f"5 index {i} Conf {conf_multi3[i]:.3f} Precision {precision_multi3[i]:.3f} recall {recall_multi3[i]:.3f}")



#Plotting lines
fig = plt.figure()
ax = plt.subplot(111)
# ax.plot(recall_day, precision_day,  marker = "o", linestyle="solid", color = "blue", markersize=0, linewidth=1, 
	# label= "1"
	# )
ax.plot(recall_night, precision_night,  marker = "o", linestyle="solid", color = "red", markersize=0, linewidth=1, 
	label= "RGB"
	)
# ax.plot(recall_multi, precision_multi,  marker = "o", linestyle="solid", color = "grey", markersize=0, linewidth=1, 
	# label= "3"
	# )
ax.plot(recall_multi2, precision_multi2,  marker = "o", linestyle="solid", color = "green", markersize=0, linewidth=1, 
	label= "thermal"
	)
ax.plot(recall_multi3, precision_multi3,  marker = "o", linestyle="solid", color = "violet", markersize=0, linewidth=1, 
	label= "multispectral"
	)



#plot diamond for marking the same value for both precision and recall
# ax.plot(best_recall_day, best_precision_day,  marker = "D", linestyle="solid", color = "k", markersize=10, linewidth=1, label=round(best_recall_day,3))
ax.plot(best_recall_night, best_precision_night,  marker = "D", linestyle="solid", color = "#CCCC00", markersize=10, linewidth=1, label= round(best_recall_night,3))
# ax.plot(best_recall_multi, best_precision_multi,  marker = "D", linestyle="solid", color = "silver", markersize=10, linewidth=1, label=round(best_recall_multi,3))
ax.plot(best_recall_multi2, best_precision_multi2,  marker = "D", linestyle="solid", color = "teal", markersize=10, linewidth=1, label=round(best_recall_multi2,3))
ax.plot(best_recall_multi3, best_precision_multi3,  marker = "D", linestyle="solid", color = "tomato", markersize=10, linewidth=1, label=round(best_recall_multi3,3))

# ax.plot(ax.get_xlim(),ax.get_ylim(), 'red', linewidth=1, label="Best model") #add diagonal line (not working well)
# ax.set_title(f"Conf {best_conf:.3f} Precision {best_precision:.3f} Recall {best_recall:.3f}")
ax.axes.set_ylim(top=1, auto=True)
ax.axes.set_xlim(left=0, right=1, auto=True) #for adjusting the fraction of the lines
# ax.tick_params(axis='y',which='major', h=10)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)

ax.set_xlabel("Precision")
ax.set_ylabel("Recall")

# plt.subplots_adjust(hspace = 0.5, wspace = 0.5) #range among plots
fig.suptitle("PR curve Night")
plt.show()
	