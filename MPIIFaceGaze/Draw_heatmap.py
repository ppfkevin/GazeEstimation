# # %matplotlib inline
#
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import cv2
#
# f, (ax1,ax2) = plt.subplots(figsize = (6,4),nrows=2)
# ax1.set_title('cubehelix map')
# ax1.set_xlabel('')
# ax1.set_xticklabels([]) #设置x轴图例为空值
# ax1.set_ylabel('kind')
#
# cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
# data = cv2.imread('F:\images\SJTUGaze\Pang_data\P02\Eyetracking\GP1\eye\p02_1.jpg', cv2.IMREAD_GRAYSCALE)
# beatmap = sns.heatmap(data, cbar=False, ax = ax1)
# with sns.color_palette('RdBu'):
#     plt.show()
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import cv2
data = cv2.imread('F:\images\SJTUGaze\Pang_data\P02\Eyetracking\GP1\eye\p02_1.jpg', cv2.IMREAD_GRAYSCALE)
z = (np.random.rand(9000000)+np.linspace(0,1, 9000000)).reshape(3000, 3000)
x, y = np.random.rand(10), np.random.rand(10)
plt.imshow(data, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
        cmap=cm.hot, norm=LogNorm())
plt.colorbar()
plt.show()