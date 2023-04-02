import cv2
import math
import numpy as np

# 摄像头内参矩阵
mtx = np.array([[805.91046546,   0.,         337.6621623 ],
                [  0.,         810.47971559, 189.95261068],
                [  0.,           0.,           1.        ]])

# 摄像头畸变系数
dist = np.array([[-2.88253157e-02,1.45975367e+00,-7.73522224e-03,8.68886757e-03,-8.05358196e+00]])

# 待测物的实际宽度单位毫米
real_width = 297

# 待测物的实际高度单位毫米
real_height = 210

# 相机焦距
focal_length = 805.9104654589185

cap = cv2.VideoCapture(0)  # 打开默认摄像头

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()  # 读取一帧视频
    dst = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度值图
    gaussian = cv2.GaussianBlur(gray, (5, 5), 0, 0)  # 高斯去噪
    ret, binary = cv2.threshold(gaussian, 127, 255, cv2.THRESH_BINARY)  # 转为二值图
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓

    max_rect = None
    max_rect_area = 0
    for cnt in contours:
        # 如果不是矩形，则忽略
        approx = cv2.approxPolyDP(cnt, 0.05*cv2.arcLength(cnt, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        # 计算矩形的面积并筛选
        rect_area = cv2.contourArea(cnt)
        if rect_area > max_rect_area:
            max_rect_area = rect_area
            max_rect = approx.astype(np.float32)

    if max_rect is not None:
        cv2.drawContours(dst, [max_rect.astype(np.int32)], -1, (0, 0, 255), 3)  # 绘制轮廓

        # 计算矩形的高度和宽度
        x_len = np.linalg.norm(max_rect[0] - max_rect[1])
        y_len = np.linalg.norm(max_rect[1] - max_rect[2])

        # 计算距离
        distance = (focal_length * real_width) / y_len

        # 在图像上显示矩形的高度、宽度和距离
        cv2.putText(dst, "distance: {:.2f}mm".format(distance), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('frame', dst)  #

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
