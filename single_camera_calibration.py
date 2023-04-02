import numpy as np
import cv2
import glob

# 棋盘格尺寸，单位为毫米
square_size = 20.0

# 棋盘格角点数
pattern_size = (9, 6)

# 设置要采用的算法
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 生成棋盘格的坐标
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# 存储棋盘格角点的世界坐标和图像坐标对
objpoints = []  # 存储棋盘格角点的世界坐标
imgpoints = []  # 存储棋盘格角点的图像坐标

# 获取所有标定图片的路径
images = glob.glob('img/calibration*.jpg')

# 依次对每张图片进行标定
for fname in images:
    # 获取当前图片
    img = cv2.imread(fname)

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # 亚像素级角点精确化
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 绘制棋盘格角点
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)

        # 存储棋盘格角点的世界坐标和图像坐标对
        objpoints.append(objp)
        imgpoints.append(corners2)

        # 获取相机内部参数和外部参数
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print(f"{fname} 标定完成")

        # 显示检测结果
        cv2.imshow('img', img)
        cv2.waitKey(500)

    else:
        print(f"{fname} 未能检测到棋盘格角点")

# 关闭窗口
cv2.destroyAllWindows()

# 获取相机内部参数和外部参数
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)



# 计算相机的焦距
focal_length = mtx[0][0]
# fy = mtx[1][1]

# print("相机内部参数：\n", mtx)
# print("畸变参数：\n", dist)
# print("相机的焦距fx：", focal_length)
#print("相机的焦距fy：",fy)
f = open('data.txt', 'w')
f.write("mtx：\n" + str(mtx) + "\n")
f.write("dist：\n" + str(dist) + "\n")
f.write("focal_length：" + str(focal_length) + "\n")
f.close()