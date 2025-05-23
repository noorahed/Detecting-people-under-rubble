import torch
import torchvision.ops as ops
import cv2

def fuse_detections(thermal_out, rgb_out, iou_threshold=0.4):
    t_boxes = thermal_out.boxes.xyxy
    r_boxes = rgb_out.boxes.xyxy
    t_conf  = thermal_out.boxes.conf
    r_conf  = rgb_out.boxes.conf

    if t_conf.numel() == 0:
        print("No thermal detections — using RGB only")
        return r_boxes, r_conf

    if r_conf.numel() == 0:
        print("No RGB detections — using thermal only")
        return t_boxes, t_conf

    iou_matrix = ops.box_iou(r_boxes, t_boxes)
    used_thermal = torch.zeros(len(t_boxes), dtype=torch.bool)

    fused_boxes = []
    fused_scores = []
    sources = []

    var_t = torch.var(t_conf)
    var_r = torch.var(r_conf)
    w_t = var_t / (var_t + var_r + 1e-6)
    w_r = var_r / (var_t + var_r + 1e-6)

    t_min, t_max = t_conf.min(), t_conf.max()
    r_min, r_max = r_conf.min(), r_conf.max()

    for i, ious in enumerate(iou_matrix):
        max_iou, t_idx = ious.max(0)
        rgb_box = r_boxes[i]
        if max_iou >= iou_threshold:
            used_thermal[t_idx] = True
            t_score = t_conf[t_idx]
            r_score = r_conf[i]
            t_norm = (t_score - t_min) / (t_max - t_min + 1e-6)
            r_norm = (r_score - r_min) / (r_max - r_min + 1e-6)
            fused_score = w_t * t_norm + w_r * r_norm
            sources.append('fused')
        else:
            fused_score = r_conf[i]
            sources.append('rgb')
        fused_boxes.append(rgb_box)
        fused_scores.append(fused_score)

    for i, used in enumerate(used_thermal):
        if not used:
            fused_boxes.append(t_boxes[i])
            fused_scores.append(t_conf[i])
            sources.append('thermal')

    return torch.stack(fused_boxes), torch.stack(fused_scores)


def draw_fused(rgb_img, boxes, scores):
    out = rgb_img.copy()
    for (x1, y1, x2, y2), s in zip(boxes.tolist(), scores.tolist()):
        if torch.isnan(torch.tensor(s)):
            continue  # Skip drawing for NaN scores

        red = int(255 * (1 - s))   # fades from 255 to 0
        green = 255
        blue = 0
        color = (blue, green, red)

        label = f"{s:.2f}"
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(out, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
    return out
