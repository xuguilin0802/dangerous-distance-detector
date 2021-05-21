# 用法
# python dangerous_distance_detector.py --input pedestrians.mp4
# python dangerous_distance_detector.py --input pedestrians.mp4 --output output.avi

# 引包
from pyimagesearch import dangerous_distancing_config as config
from pyimagesearch.detection import detect_people
from pyimagesearch.detection_car import detect_car
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# 构建命令解释
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())



# 引入COCO类名和训练好的yolo模型
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# 导出YOLO权重路径和模型配置
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
print(weightsPath)
print(configPath)

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg","yolo-coco/yolov3.weights")
# 是否用GPU加速
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# 只需要YOLO输出层
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# 视频流初始化和输出文件配置
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# 循环播放视频流中的帧
while True:
	# 从文件中读取下一帧
	(grabbed, frame) = vs.read()

	# 抓不住帧意味着结束
	if not grabbed:
		break

	# 帧中只检测车和人
	frame = imutils.resize(frame, width=700)
	results_people = detect_people(frame, net, ln,personIdx=LABELS.index("person"))
	results_car = detect_car(frame,net,ln,carIdx=LABELS.index("car"))
	# 初始化违反临界安全距离的set
	violate_people = set()
	violate_car = set()





	# 保证车>0,人>0，且人车总数>=2
	if len(results_people)+len(results_car) >= 2 and len(results_people) > 0 and len(results_car) > 0:
		# 从结果中提取质心，并计算彼此之间的欧几里得距离
		centroids_people = np.array([r[2] for r in results_people])
		centroids_car = np.array([r[2] for r in results_car])
		D = dist.cdist(centroids_people, centroids_car, metric="euclidean")




		# 遍历距离矩阵的上三角
		for i in range(0, D.shape[0]):
			for j in range(0, D.shape[1]):
				# 检查任意两个质心对之间的距离是否小于配置的像素数
				if D[i, j] < config.MIN_DISTANCE:
					# 用质心对的索引更新违规集
					violate_people.add(i)

	start_people = set()
	end_car = set()
	# 循环结果
	for (i, (prob, bbox, centroid)) in enumerate(results_people):
		# 提取人边界框和质心坐标，初始化注释颜色
		(startX, startY, endX, endY) = bbox
		# print("people")
		# print(abs((endX-startX)*(startY-endY)))
		# if abs((endX-startX)*(startY-endY))<config.MIN_PEOPLE:
		# 	if i in violate_people:
		# 		violate_people.remove(i)
		# 	continue
		(cX, cY) = centroid
		start_people.add((cX, cY))
		color = (0, 255, 0)

		# 如果索引对存在于冲突集中，那么就更新颜色
		if i in violate_people:
			color = (0, 0, 255)

		# 绘制围绕人的边界框和人的质心坐标
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)

	for (i, (prob, bbox, centroid)) in enumerate(results_car):
		# 提取车边界框和质心坐标，初始化注释颜色
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		end_car.add((cX, cY))
		# if abs((endX-startX)*(startY-endY))<config.MIN_CAR:
		# 	continue
		color = (255, 0, 0)

		# 如果索引对存在于冲突集中，那么就更新颜色
		# if i in violate_car:
		# 	color = (0, 0, 255)

		# 绘制围绕车的边界框和人的质心坐标
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)

	# 画线
	for i in start_people:
		for j in end_car:
			point_color = (255, 255, 0)  # BGR
			thickness = 1
			lineType = 4
			cv2.line(frame,i,j, point_color, thickness, lineType)


	# 在输出帧上绘制临界安全距离违规的总数
	text = "Number of people at high risk: {}".format(len(violate_people))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# 检查输出帧是否应该显示在我们的屏幕上
	if args["display"] > 0:
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# 如果输出视频文件路径已提供，而视频编写器尚未初始化，请现在进行初始化
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# 如果视频编写器不是 None，则将帧写入输出视频文件
	if writer is not None:
		writer.write(frame)