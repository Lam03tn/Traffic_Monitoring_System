import os
import cv2
import math
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np

# Bản đồ lớp sang ký tự
label_map = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
    35: 'Z'
}

def label_to_char(cls_id):
    return label_map.get(cls_id, '?')

def draw_boxes_only_labels(image, result):
    img = image.copy()
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        label = label_to_char(cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return img

def preprocess_plate_image(image):
    # 0. Resize về chiều rộng tối thiểu
    h, w = image.shape[:2]
    if w < 640:
        scale = 640 / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bước 5: Làm mượt nhẹ
    # 2. Normalize
    normalized = cv2.normalize(gray.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    normalized = (normalized * 255).astype('uint8')

    # 3. Median Blur
    denoised = cv2.medianBlur(normalized, 3)

    # 4. Threshold
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Erode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(denoised, kernel, iterations=1)


    # # # # --- NEW: Find largest contour (biển số) ---
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            # Tìm được khung 4 điểm
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            aligned = four_point_transform(image, rect)
            return aligned  # Trả về ảnh đã được căn thẳng

    # Nếu không tìm được contour phù hợp, trả về ảnh đã xử lý đơn giản
    result_bgr = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
    return result_bgr


def order_points(pts):
    """Sắp xếp các điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def four_point_transform(image, pts):
    """Biến đổi phối cảnh 4 điểm về ảnh phẳng"""
    (tl, tr, br, bl) = pts

    # Chiều rộng mới
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    # Chiều cao mới
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # Ma trận đích
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Tính ma trận biến đổi
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def linear_equation(x1, y1, x2, y2):
    if x2 == x1:
        return None, None
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    if a is None:
        return False
    y_pred = a * x + b
    return math.isclose(y_pred, y, abs_tol=3)

def iou(box1, box2):
    # Tính IoU giữa hai hộp [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def read_plate(char_model, plate_crop):
    LP_type = "1"  # Mặc định là 1 dòng
    char_result = char_model.predict(
        source=plate_crop,
        conf=0.3,
        device='cpu',
        iou=0.4,
        verbose=False
    )
    result = char_result[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    class_names = result.names

    if len(boxes) < 7 or len(boxes) > 12:
        return "unknown", char_result

    # Kết hợp thông tin box, label, score
    detections = []
    for i in range(len(boxes)):
        detections.append({
            "box": boxes[i],
            "score": scores[i],
            "label": int(labels[i])
        })

    # Sắp xếp theo score giảm dần để ưu tiên giữ lại box tốt
    detections.sort(key=lambda x: -x["score"])

    # Non-overlapping filter
    final_detections = []
    for det in detections:
        overlap = False
        for kept in final_detections:
            if iou(det["box"], kept["box"]) > 0.1:
                overlap = True
                break
        if not overlap:
            final_detections.append(det)

    # Trích xuất trung tâm và nhãn
    center_list = []
    y_sum = 0
    for det in final_detections:
        x1, y1, x2, y2 = det["box"]
        x_c = (x1 + x2) / 2
        y_c = (y1 + y2) / 2
        label = class_names[det["label"]]
        y_sum += y_c
        center_list.append([x_c, y_c, label])

    if len(center_list) < 7 or len(center_list) > 12:
        return "unknown", char_result

    # Kiểm tra xem biển 1 dòng hay 2 dòng
    l_point = min(center_list, key=lambda p: p[0])
    r_point = max(center_list, key=lambda p: p[0])

    for ct in center_list:
        if l_point[0] != r_point[0]:
            if not check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]):
                LP_type = "2"
                break

    y_mean = y_sum / len(center_list)
    margin = 8
    line_1 = []
    line_2 = []
    license_plate = ""

    if LP_type == "2":
        for c in center_list:
            if c[1] > y_mean + margin:
                line_2.append(c)
            elif c[1] < y_mean - margin:
                line_1.append(c)
            else:
                line_1.append(c)
        for l1 in sorted(line_1, key=lambda x: x[0]):
            license_plate += str(l1[2])
        license_plate += "-"
        for l2 in sorted(line_2, key=lambda x: x[0]):
            license_plate += str(l2[2])
    else:
        for l in sorted(center_list, key=lambda x: x[0]):
            license_plate += str(l[2])

    return license_plate, char_result

def changeContrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def compute_skew(src_img, center_thres):
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')
    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 1.5, maxLineGap=h/3.0)
    if lines is None:
        return 1

    min_line = 100
    min_line_pos = 0
    for i in range (len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            center_point = [((x1+x2)/2), ((y1+y2)/2)]
            if center_thres == 1:
                if center_point[1] < 7:
                    continue
            if center_point[1] < min_line:
                min_line = center_point[1]
                min_line_pos = i

    angle = 0.0
    nlines = lines.size
    cnt = 0
    for x1, y1, x2, y2 in lines[min_line_pos]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30: # excluding extreme rotations
            angle += ang
            cnt += 1
    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi

def deskew(src_img, change_cons, center_thres):
    if change_cons == 1:
        return rotate_image(src_img, compute_skew(changeContrast(src_img), center_thres))
    else:
        return rotate_image(src_img, compute_skew(src_img, center_thres))


def process_license_plate(plate_model_path, char_model_path, video_path, output_dir="plates"):
    plate_model = YOLO(plate_model_path)
    char_model = YOLO(char_model_path)
    os.makedirs(output_dir, exist_ok=True)

    results = plate_model.predict(
        source=video_path,
        save=False,
        stream=True,
        device='cpu'
    )

    for i, result in enumerate(results):
        if(i % 3 != 0):
            continue

        frame = result.orig_img.copy()
        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            filename = f"plate_high_reso_{i:03d}_{j+1}.jpg"

            # h, w = plate_crop.shape[:2]
            # if w < 640:
            #     scale = 640 / w
            #     plate_crop = cv2.resize(plate_crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

            processed_plate = preprocess_plate_image(plate_crop)

            # Thử các cấu hình xoay và chọn kết quả tốt nhất
            best_license_plate = "unknown"
            best_char_result = None
            best_conf_score = 0.0

            # Thử các tổ hợp cc (change contrast) và ct (center threshold)
            for cc in range(0, 2):
                for ct in range(0, 2):
                    # Xoay ảnh
                    rotated_plate = deskew(processed_plate, cc, ct)
                    # Nhận diện ký tự
                    license_plate_text, char_result = read_plate(char_model, rotated_plate)
                    
                    # Lấy confidence scores từ kết quả dự đoán
                    conf_scores = char_result[0].boxes.conf.cpu().numpy()
                    # Tính tổng confidence score (hoặc trung bình nếu muốn)
                    total_conf_score = conf_scores.sum() / len(conf_scores) if len(conf_scores) > 0 else 0.0

                    # Cập nhật kết quả tốt nhất nếu confidence score cao hơn và không phải "unknown"
                    if license_plate_text != "unknown" and total_conf_score > best_conf_score:
                        best_conf_score = total_conf_score
                        best_license_plate = license_plate_text
                        best_char_result = char_result

            # Sử dụng kết quả tốt nhất
            if best_char_result is not None:
                img_with_labels = draw_boxes_only_labels(processed_plate, best_char_result[0])
                plt.figure(figsize=(8, 3))
                plt.imshow(cv2.cvtColor(img_with_labels, cv2.COLOR_BGR2RGB))
                plt.title(f"Frame {i} - Predicted: {best_license_plate} (Conf: {best_conf_score:.2f})")
                plt.axis("off")
                plt.savefig(os.path.join(output_dir, f"annotated_{filename.replace('.jpg', '.png')}"))
                plt.close()

if __name__ == "__main__":
    model_plate_path = r"C:\Users\LamNH\Desktop\assignment\ĐATN\Traffic-system\model\plate detection\plate_detection_v2_choose.pt"
    model_char_path = r"C:\Users\LamNH\Desktop\assignment\ĐATN\Traffic-system\model\character detection\character_detection_v13.pt"
    video_path = r"C:\Users\LamNH\Desktop\assignment\ĐATN\Traffic-system\Camera_Simulator_Stream\videos\cam1.mp4"
    process_license_plate(model_plate_path, model_char_path, video_path)