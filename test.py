from ultralytics import YOLO
import torch

# 检查GPU可用性
print("GPU available: ", torch.cuda.is_available())

# 加载训练好的模型
checkpoint = "runs/detect/train2/weights/best.pt"
input_image = "/root/yolov12/datas/test/images"
model = YOLO(checkpoint)

# 运行检测
results = model(source=input_image, conf=0.3, show=True, save=True)

# 打印结果
print('Results: ', results)
print('Boxes: ', results[0].boxes)
print('Done!')