# dangerous-distance-detector
## Using YOLOv3,OpenCV and Tensorflow accessing pedestrian hazards

This is my Undergraduate Graduation Design，it can access pedestrian hazards by using object detection. The main idea of algorithm is actually sample,it is pixel distance which judges the pedestrian hazards. You can change the ''MIN_DISTANCE'' which in "pyimagesearch/dangerous_distancing_config" to map to the actual distance. Here is the demo:

![图10](F:\毕业设计\图片\图10.png)

![图11](F:\毕业设计\图片\图11.png)

The Box of person will turn to red when the pixel distance between person and car less than the

 ''MIN_DISTANCE'' .

### 	Environment

| **工具包**    | **版本号** |
| ------------- | ---------- |
| imutils       | 0.5.4      |
| numpy         | 1.18.5     |
| opencv-python | 4.3.0.38   |
| scipy         | 1.6.1      |
| tensorflow    | 1.15.0     |

### Downloading official pretrained weights

 Download the yolov3 weights by clicking [here](https://pjreddie.com/media/files/yolov3.weights) or search it . Put it in "yolo-coco" file.

 ### Running

Choosing your video in main file.

```
python dangerous_distance_detector.py --input video_detrac_1.mp4 --output out_video_detrac_1_pt.avi
```

### Example

![demo](F:\毕业设计\图片\demo.gif)

