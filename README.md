# OBJECT-DETECTION-AND-SORTING-ON-THE-CONVEYOR-SYSTEM

In this project, our goal is to improve the efficiency and accuracy of a conveyor system by implementing an intelligent object detection and sorting system. We aim to identify and sort objects running on the conveyor belt in real-time with 100% accuracy.
To achieve this, we will use a FINGERS 1080-high-resolution webcam placed above the conveyor belt, capturing images of the objects as they move through the system. The conveyor belt is motor-driven, ensuring smooth movement, and encoders will measure object speed and position for precise synchronization.
Using advanced deep learning techniques, like the YOLO (You Only Look Once) model, we will perform real-time object detection, focusing initially on differentiating between boxes and bottles. The webcam will only capture the conveyor belt and objects, minimizing background noise.
Our approach will prioritize computational efficiency while maintaining high accuracy. We will address variations in object orientation, shape, size, and color for robust detection and sorting.
By integrating this system with a parallel robot, we will enable precise timing and positioning of objects, ensuring efficient sorting.
Overall, this project aims to enhance the conveyor system's capabilities, providing reliable and fast object detection and sorting, thereby improving industrial automation processes.

# Additional Info

Training Dataset size: 15,517 Validation Dataset size: 61 which have box and bottle images from OpenImagesv7 and webcam images on the conveyor system.

During the training process of the YOLOv8m model, the initial configuration was set to run for 300 epochs, and each epoch had a batch size of 16, resulting in a total of 948 batches for each epoch. However, the training process early stopped at the 104th epoch, which means the model did not complete all 300 epochs as originally planned.
Throughout the 104 epochs of training, the graph below shows the progress of our YOLOv8m model by tracking various metrics. These metrics included the training and validation losses for bounding box regression (box loss), class prediction (cls loss), and object detection confidence loss (dfl loss). We also measured the Mean Average Precision at 50 IoU (mAP50) and Mean Average Precision from 50 to 95 IoU (mAP50-95) at each epoch.

# Test.py File

Modify and run the test.py file to run the custom trained model (best.pt has the weights) on your our system by choosing suitable webcam for working.

# yolov8_labels.py File

If you are using your own custom dataset to train your yolo model you need to make the labels yolov8 compatible if it isn't already, this py file will help to make it yolov8 compatible.
*Note : this code specifically deals with only two classes box and bottle though with few changes in the code it can be generalized.

