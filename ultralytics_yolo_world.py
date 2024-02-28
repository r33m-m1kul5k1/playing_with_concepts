from ultralytics import YOLOWorld

# Initialize a YOLO-World model
model = YOLOWorld('yolov8l-world.pt')  # or select  for different sizes

model.set_classes(["car", "building"])


# Execute inference with the YOLOv8s-world model on the specified image
# /home/mafat/Desktop/VisDrone2019-VID-train/sequences/uav0000270_00001_v/0000004.jpg
# /home/mafat/Downloads/renders/(-200, -120).png
results = model.predict('/home/mafat/Downloads/renders/(-200, -120).png', conf=0.25)

# Show results
results[0].show()