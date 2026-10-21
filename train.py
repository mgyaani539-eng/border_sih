from ultralytics import YOLO

# Load base model
model = YOLO('yolov8n.pt')

# Point directly to data.yaml using its absolute path
results = model.train(
    data=r'C:\Users\M RAMA MANI\OneDrive\Desktop\border_ai_sih\dataset\data.yaml', 
    epochs=10,        
    imgsz=640,        
    batch=8           
)

print("Training finished! Your custom model is ready.")