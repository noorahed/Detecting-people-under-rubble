from ultralytics import YOLO

def load_models():
    thermal_model = YOLO('disaster_detection/weights/Therm2_best.pt')
    rgb_model = YOLO('disaster_detection/weights/RGB2_best.pt')
    return thermal_model, rgb_model
