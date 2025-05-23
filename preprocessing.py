import torchvision.transforms as T
import cv2


def resize_to_nearest_stride(image, stride=32):
    h, w = image.shape[:2]
    new_h = (h // stride) * stride
    new_w = (w // stride) * stride
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def preprocess_rgb(frame):
    resized = resize_to_nearest_stride(frame)
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor()
    ])
    tensor = transform(resized).unsqueeze(0)  # (1, 3, H, W)
    return tensor

def preprocess_thermal(frame):
    resized = resize_to_nearest_stride(frame)
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor()
    ])
    tensor = transform(resized).unsqueeze(0)
    return tensor
