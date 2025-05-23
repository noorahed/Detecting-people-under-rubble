import cv2
from disaster_detection.models import load_models
from disaster_detection.preprocessing import preprocess_thermal, preprocess_rgb
from disaster_detection.inference import parallel_inference
from disaster_detection.fusion import fuse_detections, draw_fused

def live_video_detection(source=0):
    # Open video stream (0 = default webcam, or path to video file / IP stream)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Load models once
    thermal_model, rgb_model = load_models()


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or failed to grab frame.")
            break


        # Simulate thermal image (in real case, replace with actual thermal capture)
        thermal_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal_frame = cv2.cvtColor(thermal_frame, cv2.COLOR_GRAY2BGR)  # Make it 3-channel for compatibility

        # Preprocess frames
        t_tensor = preprocess_thermal(thermal_frame)
        r_tensor = preprocess_rgb(frame)

        # Inference
        t_out, r_out = parallel_inference(thermal_model, rgb_model, t_tensor, r_tensor)

        # Fuse results
        boxes, fused_scores = fuse_detections(t_out, r_out)

        # Draw results
        annotated = draw_fused(frame, boxes, fused_scores)

        # Show frame
        cv2.imshow("Live Fused Detection", annotated)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()