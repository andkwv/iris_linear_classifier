import csv
import random as rd
import math as mt


# libraries for the graph only
import matplotlib.pyplot as plt

# with open('D:\KULIAH\PELAJARAN_SEMESTER_6\Machine Learning\iris_dataset.csv', newline='')  as csvfile:
# 	 datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
# 	 for row in datareader:
# 	 	print(row)

# Here a class is used to load the file once instead of reading it over and over again, for effieciency
class IrisData():
	def __init__(self, filename):
		with open(filename, "r") as f_input:
			csv_input = csv.reader(f_input)
			self.details = list(csv_input)

	def get_col_row(self, col, row):
		return self.details[row-1][col-1] 
		# Python index starts from 0 so we have to substract by 1

data = IrisData("D:\KULIAH\PELAJARAN_SEMESTER_6\Machine Learning\iris.csv")

# sepal_length = data.get_col_row(1,1)
# sepal_width = data.get_col_row(2,1)
# petal_length = data.get_col_row(3,1)
# petal_width = data.get_col_row(4,1)
# y1 = data.get_col_row(6,1)
# y2 = data.get_col_row(7,1)


# print("%s %s %s %s %s %s" % (sepal_length, sepal_width, petal_length, petal_width, y1, y2))

# the function to retrieve values from the csv file
def getSepalPetal(i):
	sepal_length = float(data.get_col_row(1,i))
	sepal_width = float(data.get_col_row(2,i))
	petal_length = float(data.get_col_row(3,i))
	petal_width = float(data.get_col_row(4,i))

	y1 = float(data.get_col_row(6,i))
	y2 = float(data.get_col_row(7,i))

	# assigned to a list to simplify process 
	SepalPetal_list = [sepal_length, sepal_width, petal_length, petal_width, y1, y2]
	return SepalPetal_list
	# we simply return the list

# the values will be pre calculated before calling the target calculation function
def target(t1, t2, t3, t4, b):
	result = t1 + t2 + t4 + b
	return result

# we calculate the sigmoid simply using the formula
def sigmoid(targetx):
	exp = mt.exp(-targetx)
	sgmd = 1 / (1 + exp)
	return sgmd

# we normalize, if it is below 0.5 then we will predict it to be 1
def prediction(sigmoid_var):
	if(sigmoid_var < 0.5):
		return 0
	else:
		return 1

def error(sigmoid_var, predc):
	err = (abs(sigmoid_var - predc)) ** 2
	return err

def dtheta(sigmoid_var, y_target, x_theta):
	dtheta_res = 2*(sigmoid_var - y_target)*(1 - sigmoid_var) * sigmoid_var * x_theta
	return dtheta_res

#for the initial values of the theta we will use a random number
theta_1 = rd.random()
theta_2 = rd.random()
theta_3 = rd.random()
theta_4 = rd.random()
bias_a = rd.random()

theta_5 = rd.random()
theta_6 = rd.random()
theta_7 = rd.random()
theta_8 = rd.random()
bias_b = rd.random()

answer = input("What do you want the learning rate to be ? \na. 0.1 	b. 0.8\n")
if(answer == 'a'):
	l_rate = 0.1
else:
	l_rate = 0.8


# we loop for each epoch, in this case we require 100 epochs.
for epoch in range(1,101):

	# Here we will then loop for each row
	#before we begin we define for each epoch the error variable
	epoch_err_1 = 0		#value to store the epoch
	epoch_err_2 = 0 	

	accuracy = 0		#value to store accuracy
	accuracy_1 = 0
	accuracy_2 = 0

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	for i in range(1,151): #Here we start from 1, adjusting to our class function (-1)
		SP = getSepalPetal(i) #we call the function to retrieve value FOR THE CURRENT ITERATION/ROW

	# **************************************************************************************************
		# we pre-calculate to simplify calculating target and others
		r1 = theta_1 * SP[0]		
		r2 = theta_2 * SP[1]
		r3 = theta_3 * SP[2]
		r4 = theta_4 * SP[3]
		y1 = SP[4]

		#to calculate each of the values we call the function
		target_1 = target(r1,r2,r3,r4, bias_a)
		sigmoid_1 = sigmoid(target_1)
		prediction_1 = prediction(sigmoid_1)
		err_1 = error(sigmoid_1, y1)

		# Now we generate the dtheta to improve each theta
		dt_list_1 = []
		for j in range(0,4):		
			dt_list_1.append(dtheta(sigmoid_1, y1, SP[j]))
		# We begin by calculating dtheta for each dtheta, then we calculate for the bias	
		dt_list_1.append(dtheta(sigmoid_1, y1, 1))


	# *******************************************************************************************************
		
		#prepare for second prediction

		r5 = theta_5 * SP[0]
		r6 = theta_6 * SP[1]
		r7 = theta_7 * SP[2]
		r8 = theta_8 * SP[3]
		y2 = SP[5]

		target_2 = target(r1,r2,r3,r4, bias_b)
		sigmoid_2 = sigmoid(target_2)
		prediction_2 = prediction(sigmoid_2)
		err_2 = error(sigmoid_2, y2) 

		dt_list_2 = []
		for k in range(4):		
			dt_list_2.append(dtheta(sigmoid_2, y2, SP[k]))
		dt_list_2.append(dtheta(sigmoid_2, y2, 1))

	# **********************************************************************************************************

		#NOW WE CALCULATE THE VALUES FOR THE NEXT ITERATION THETA (IMPROVED FROM DTHETA)

		theta_1 =  theta_1 - (l_rate * dt_list_1[0])
		theta_2 =  theta_2 - (l_rate * dt_list_1[1])
		theta_3 =  theta_3 - (l_rate * dt_list_1[2])
		theta_4 =  theta_4 - (l_rate * dt_list_1[3])
		bias_a =  bias_a - (l_rate * dtheta(sigmoid_1, y1, 1))

		theta_5 =  theta_5 - (l_rate * dt_list_2[0])
		theta_6 =  theta_6 - (l_rate * dt_list_2[1])
		theta_7 =  theta_7 - (l_rate * dt_list_2[2])
		theta_8 =  theta_8 - (l_rate * dt_list_2[3])
		bias_b =  bias_b - (l_rate * dtheta(sigmoid_2, y2, 1))

	# **********************************************************************************************************
		#print("Error 1: %s '\t' Error 2: %s" % (err_1, err_2)) #for testing what each epoch looks like
		epoch_err_1 += err_1 
		epoch_err_2 += err_2 

		if(prediction_1 == y1):
			accuracy_1 += 1

		if(prediction_2 == y2):
			accuracy_2 += 1

		if(prediction_1 == y1 and prediction_2 == y2):
			accuracy += 1

		#print(accuracy_1, accuracy_2, accuracy)

	# Finally at the end of each epoch we calculate the error rate
	err_avg_1 = epoch_err_1/150
	err_avg_2 = epoch_err_2/150
	
	accuracy_1 = accuracy_1/150
	accuracy_2 = accuracy_2/150
	accuracy = accuracy/150

	print("Epoch %s. \nError_avg 1: %s \t Error_avg 2: %s" % (epoch, err_avg_1, err_avg_2))

	print("Accuracy_avg_1: %s Accuracy_avg_2: %s Accuracy: %s" % (accuracy_1, accuracy_2, accuracy))
	# print("Accuracy_avg: %s" % (accuracy))
	print("\n\n")

	plt.figure(1)
	plt.plot(epoch, err_avg_1, '-o')
	
	plt.figure(2)
	plt.plot(epoch, err_avg_2, '-o')
	
	plt.figure(3)
	plt.plot(epoch, accuracy, '-o')

print("THIS IS RESULT WITH %s LEARNING RATE" % (l_rate))
plt.show()

