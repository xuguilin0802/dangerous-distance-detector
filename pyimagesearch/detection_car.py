# 引入需要使用的包
from .dangerous_distancing_config import NMS_THRESH
from .dangerous_distancing_config import MIN_CONF
from .dangerous_distancing_config import MIN_CAR
import numpy as np
import cv2


def detect_car(frame,net,ln,carIdx=2):
# 获取视频帧的长和宽
    (H,W)= frame.shape[:2]
    results = []

# 构建blob作为网络的输入，将blob网络，网络前向计算得到边界框和相应的概率
    blob = cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)#yolov3有三个输入层，所以layerOutpus的长度为3，包含每个输入层的检测结果

# 初始化边界框、中心点、置信度课表

    boxes = []
    centroids = []
    confidences = []

# 循环网络每个输出层

    for output in layerOutputs:
        # 循环每个输出层输出的每个检测边界框
        for detect_car in output:
            # 提取预测边界框的类别、对应的分数
            scores = detect_car[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # 筛选边界框：保留类别为车的边界框且置信度大于阈值的边界框
            if classID == carIdx and confidence > MIN_CONF:

                # yolo网络的输出边界坐标为 (centerX,centerY,width,height),将网络输出的边界框坐标改为(左上x,左上y,w,h)
                box = detect_car[0:4] * np.array([W,H,W,H])
                (centerX,centerY,width,height) = box.astype("int")
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                #更新刚才创建的列表：边界框列表、中心点列表、置信度列表
                boxes.append([x,y,int(width),int(height)])
                centroids.append((centerX,centerY))
                confidences.append(float(confidence))

    # 非极大抑制
    idxs = cv2.dnn.NMSBoxes(boxes,confidences,MIN_CONF,NMS_THRESH)

    if len(idxs) > 0:
        # 循环每一辆车
        for i in idxs.flatten():
            # 提取边界框的坐标
            (x,y) = (boxes[i][0],boxes[i][1])
            (w,h) = (boxes[i][2],boxes[i][3])

            r = (confidences[i],(x,y,x+w,y+h),centroids[i])
            if w*h>MIN_CAR:
                results.append(r)#results列表，包含每辆检测车的置信度，每辆车的边界框，每辆车的质心

    return results