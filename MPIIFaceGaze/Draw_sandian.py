import numpy as np
import matplotlib.pyplot as plt

# N = 600
# x = np.random.randn(N)*20*np.random.choice([-1, 1])
# X = np.array([])
# for k in range(len(x)):
#     if np.abs(x[k])<25:
#         X = np.append(X, x[k])
# y = np.zeros(X.shape)
# z = np.zeros((X.shape[0], 2))
# for i in range(len(y)):
#     y[i] = X[i] + np.random.random()*np.random.choice([-1, 1])*6
#     z[i] = np.array([X[i], y[i]])
# np.savetxt('test.txt', z, fmt='%f', delimiter=',')

# data = np.loadtxt('F:\images\SJTUGaze\Pang_data\P02\Eyetracking\GP1\headpose\output-Ptest.txt', dtype = np.str)
# X = np.zeros(data.shape[0])
# y = np.zeros(data.shape[0])
# for i in range(len(data)):
#     X[i] = data[i, 2].astype('float')
#     y[i] = data[i, 1].astype('float')
    # if (np.abs(X[i]) > 15) and (np.abs(X[i]) < 20):
    #     y[i] = X[i] + np.random.random() * np.random.choice([-1, 1]) * 2.4
    # elif (X[i] > -15) and (X[i] < -10):
    #     y[i] = X[i] + np.random.random() * np.random.choice([-1, 1]) * 1.8
    # elif (X[i] > -25) and (X[i] < -20):
    #     y[i] = X[i] + np.random.random() * np.random.choice([-1, 1]) * 1.8
    # else:
    #     y[i] = X[i] + np.random.random() * np.random.choice([-1, 1]) * 0.6
data1 = np.loadtxt('F:\images\SJTUGaze\Pang_data\P01\Eyetracking\GP1\headpose\output-Ptest.txt', dtype = np.str)
data2 = np.loadtxt('F:\images\SJTUGaze\Pang_data\P01\Eyetracking\GP2\headpose\output-Ptest.txt', dtype = np.str)
data3 = np.loadtxt('F:\images\SJTUGaze\Pang_data\P01\Eyetracking\GP3\headpose\output-Ptest.txt', dtype = np.str)
data4 = np.loadtxt('F:\images\SJTUGaze\Pang_data\P01\Eyetracking\GP4\headpose\output-Ptest.txt', dtype = np.str)
tmp = [data1, data2, data3, data4]
X = np.zeros(data1.shape[0] + data2.shape[0] + data3.shape[0] + data4.shape[0])
y = np.zeros(data1.shape[0] + data2.shape[0] + data3.shape[0] + data4.shape[0])
ind = 0
#yaw：-25，20 pitch：-25，15
for k in range(4):
    for i in range(len(tmp[k])):
        X[ind + i] = tmp[k][i, 1].astype('float')
        y[ind + i] = tmp[k][i, 2].astype('float')
        # if (np.abs(X[i]) > 15) and (np.abs(X[i]) < 20):
        #     y[ind] = X[ind] + np.random.random() * np.random.choice([-1, 1])
        # elif (X[i] > -15) and (X[i] < -10):
        #     y[ind] = X[ind] + np.random.random() * np.random.choice([-1, 1])
        # elif (X[i] > -25) and (X[i] < -20):
        #     y[ind] = X[ind] + np.random.random() * np.random.choice([-1, 1])
        # else:
        #     y[ind] = X[ind] + np.random.random() * np.random.choice([-1, 1])
    ind += len(tmp[k])
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
# ax1.set_title('俯仰角pitch误差对比')
ax1.set_xlabel('yaw')
ax1.set_ylabel('pitch')
plt.scatter(X, y, s=5, marker='.')
plt.show()