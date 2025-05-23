from disaster_detection.models import load_models
from disaster_detection.preprocessing import preprocess_thermal, preprocess_rgb
from disaster_detection.inference import parallel_inference
from disaster_detection.fusion import fuse_detections, draw_fused
from disaster_detection.utils import live_video_detection
import cv2
import os



def log_image_with_boxes(img, boxes, scores=None, log_name="log", out_dir="logs", color=(0,255,0)):
    os.makedirs(out_dir, exist_ok=True)
    img_copy = img.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        if scores is not None:
            score = float(scores[i])
            cv2.putText(img_copy, f"{score:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    save_path = os.path.join(out_dir, f"{log_name}.jpg")
    cv2.imwrite(save_path, img_copy)
    print(f"Logged {log_name} image to: {save_path}")


def main_pipeline(thermal_image, rgb_image):
    # Load models
    thermal_model, rgb_model = load_models()

    #  Preprocess inputs
    t_tensor = preprocess_thermal(thermal_image)
    r_tensor = preprocess_rgb(rgb_image)

    # Parallel inference
    t_out, r_out = parallel_inference(thermal_model, rgb_model, t_tensor, r_tensor)

    print("Thermal confs:", t_out.boxes.conf)
    print("RGB confs:", r_out.boxes.conf)


    # fuse and get RGB‐boxes + fused scores
    boxes, fused_scores = fuse_detections(t_out, r_out)


    log_image_with_boxes(rgb_image, r_out.boxes.xyxy, r_out.boxes.conf, log_name="rgb_raw", color=(255,0,0))
    log_image_with_boxes(thermal_image, t_out.boxes.xyxy, t_out.boxes.conf, log_name="thermal_raw", color=(0,0,255))
    log_image_with_boxes(rgb_image, boxes, fused_scores, log_name="fused_result", color=(0,255,0))


    annotated = draw_fused(rgb_image, boxes, fused_scores)

    cv2.imshow("Fused Detections", annotated)

    # Wait until any key is pressed (milliseconds; 0 = infinite)
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    return annotated #, report

def main():
    mode = input("Choose mode (image/video): ").strip().lower()

    if mode == "image":
        # Paths to images
        thermal_img_path = input("Enter path to thermal image: ").strip()
        rgb_img_path = input("Enter path to RGB image: ").strip()

        # Load images
        thermal_img = cv2.imread(thermal_img_path)
        rgb_img = cv2.imread(rgb_img_path)

        if thermal_img is None or rgb_img is None:
            raise FileNotFoundError("One or more images failed to load. Check the paths!")

        output = main_pipeline(thermal_img, rgb_img)
        print("Detection Map:", output)

    elif mode == "video":
        source = input("Enter video source (0 for webcam, or path to video file): ").strip()
        try:
            source = int(source)
        except ValueError:
            pass  # Assume it's a file path if not an integer
        live_video_detection(source)

    else:
        print("Invalid mode selected. Please choose 'image' or 'video'.")

if __name__ == '__main__':

    main()
