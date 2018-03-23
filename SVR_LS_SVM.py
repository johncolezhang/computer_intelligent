import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from liquidSVM import lsSVM
from sklearn.svm import SVR
import time
from sklearn.metrics import mean_squared_error



if __name__ == "__main__":
    point_num = 50
    # x = np.random.uniform(-5, 5, point_num).reshape((point_num, 1))
    # y = np.random.uniform(-5, 5, point_num).reshape((point_num, 1))
    # z = np.sin(np.add(x, y))
    # ax = plt.subplot(111, projection='3d')
    # ax.scatter(x, y, z, c='y')
    # ax.set_zlabel('Z')
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')
    # plt.title("scatter")
    # plt.show()
    X = np.sort(np.random.uniform(-5, 5, point_num)).reshape((point_num, 1))
    Y = np.sort(np.random.uniform(-5, 5, point_num)).reshape((point_num, 1))
    Z = np.sin(np.add(X, Y))
    X, Y = np.meshgrid(X, Y)
    Z = np.sin(np.add(X, Y))

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # ax.set_zlabel('Z')
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')
    # plt.title("curve")
    # plt.show()

    start_svr = time.clock()
    svr = SVR(kernel='rbf')
    Z_svr = svr.fit((X + Y).reshape((point_num * point_num, 1)), Z.ravel())\
        .predict((X + Y).reshape((point_num * point_num, 1)))
    stop_svr = time.clock()
    mse_svr = mean_squared_error(Z.ravel(), Z_svr.ravel())
    print("time in SVR:", stop_svr - start_svr)
    print("mse in SVR:", mse_svr)
    # ax.plot_surface(X, Y, Z_svr.reshape((point_num, point_num)), rstride=1, cstride=1, cmap=plt.get_cmap('plasma'))
    # ax.set_zlabel('Z')
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')
    # plt.title("SVR")
    # plt.show()

    start_ls = time.clock()
    Z_ls, error = lsSVM((X + Y).reshape((point_num * point_num, 1)), Z.ravel(), KERNEL='GAUSS_RBF')\
        .test((X + Y).reshape((point_num * point_num, 1)))
    stop_ls = time.clock()
    mse_ls = mean_squared_error(Z.ravel(), Z_ls.ravel())
    print("time in LS-SVM:", stop_ls - start_ls)
    print("mse in LS-SVM:", mse_ls)

    ax.plot_surface(X, Y, Z_ls.reshape((point_num, point_num)), rstride=1, cstride=1, cmap=plt.get_cmap('magma'))
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.title("LS-SVM")
    plt.show()




