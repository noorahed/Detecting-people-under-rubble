from concurrent.futures import ThreadPoolExecutor

def infer_thermal(model, tensor, conf=0.35, iou=0.4):
    output = model.predict(tensor, conf=conf, iou=iou, verbose=False)
    return output[0]

def infer_rgb(model, tensor, conf=0.35, iou=0.4):
    output = model.predict(tensor, conf=conf, iou=iou, verbose=False)
    return output[0]


def parallel_inference(thermal_model, rgb_model, thermal_tensor, rgb_tensor):
    with ThreadPoolExecutor() as executor:
        future_thermal = executor.submit(infer_thermal, thermal_model, thermal_tensor)
        future_rgb = executor.submit(infer_rgb, rgb_model, rgb_tensor)
        thermal_output = future_thermal.result()
        rgb_output = future_rgb.result()
    return thermal_output, rgb_output

