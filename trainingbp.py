import tensorlayer as tl
import tensorflow as tf
import xlrd
import numpy as np
#define nn structure params
inputunits=4
layerunits=8
outputunits=1
#read data from xls
data_eng=[]
fname="data_eng/datarow.xlsx"
bk1 = xlrd.open_workbook(fname)
sh1 = bk1.sheet_by_name("Sheet1")
data_EXP=[]
fname="data_eng/datarowEXP.xlsx"
bk2 = xlrd.open_workbook(fname)
sh2 = bk2.sheet_by_name("Sheet1")
data_IMP=[]
fname="data_eng/datarowIMP.xlsx"
bk3 = xlrd.open_workbook(fname)
sh3 = bk3.sheet_by_name("Sheet1")
data_GDP=[]
fname="data_eng/datarowGDP.xlsx"
bk4 = xlrd.open_workbook(fname)
sh4 = bk4.sheet_by_name("Sheet1")
datarow=[]
fname="data_eng/datarowPOP.xlsx"
bk5 = xlrd.open_workbook(fname)
sh5 = bk5.sheet_by_name("Sheet1")
for i in range(sh1.nrows):
	row_data1 = sh1.row_values(i)
	data_eng.append(row_data1)
	row_data2 = sh2.row_values(i)

	row_data3 = sh3.row_values(i)

	row_data4 = sh4.row_values(i)

	row_data5 = sh5.row_values(i)
	datarow.append([row_data2[0],row_data3[0],row_data4[0],row_data5[0]])
#print(data_GDP)
#data_eng=np.array(data_eng)
#data_EXP=np.array(data_EXP)
#data_IMP=np.array(data_IMP)
#data_GDP=np.array(data_GDP)
#data_POP=np.array(data_POP)
#print(data_GDP)
#x_train=[data_EXP[0:34],data_IMP[0:34],data_GDP[0:34],data_POP[0:34]]
#print(datarow[0:34])
x_train=np.array(datarow[0:34])
x_test=np.array(datarow[35:42])
#define placeholder
#None is the batchsize actually while inputunits is the inputlayer units number

x=tf.placeholder("float",[None,inputunits])
y_=tf.placeholder("float",[None,outputunits])

#define network structure with 0 weights and thresholds
W=tf.Variable(tf.zeros([inputunits,layerunits]))
b=tf.Variable(tf.zeros([outputunits]))

y=tf.matmul(x,W)+b
# use cross entropy as loss function
lossfunction=-tf.reduce_sum(y_*tf.log(y))
errorfunction=tf.reduce_mean(tf.abs(y-y_)/y_)
# train function
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(errorfunction)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(800):
	batch_xs=x_train
	batch_ys=np.array(data_eng[0:34])
	sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys}) 
	if i&100==0:
		print(sess.run(errorfunction,feed_dict={x:x_test,y_:np.array(data_eng[35:42])}))
	#print(errorfunction)
	#print(batch_ys)
	#print(sess.run(W))



sess.close()

	 




