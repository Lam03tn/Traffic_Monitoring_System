import os
import time
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import tritonclient.grpc as triton_grpc

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
    """Convert class ID to character."""
    return label_map.get(cls_id, '?')

def preprocess_plate_image(image):
    """Preprocess license plate image for detection."""
    # Resize to minimum width
    h, w = image.shape[:2]
    if w < 640:
        scale = 640 / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # return image
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize
    normalized = cv2.normalize(gray.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    normalized = (normalized * 255).astype('uint8')

    # Median Blur
    denoised = cv2.medianBlur(normalized, 3)

    # Threshold
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Erode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(denoised, kernel, iterations=1)

    # Find largest contour (license plate)
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            # Found a quadrilateral
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            aligned = four_point_transform(image, rect)
            return aligned

    # Return processed image if no suitable contour found
    result_bgr = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
    return result_bgr

def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def four_point_transform(image, pts):
    """Apply perspective transform to flatten image."""
    (tl, tr, br, bl) = pts
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def change_contrast(img):
    """Enhance image contrast using CLAHE."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def rotate_image(image, angle):
    """Rotate image by given angle."""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def compute_skew(src_img, center_thres):
    """Compute skew angle of the image."""
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        raise ValueError('Unsupported image type')

    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 1.5, maxLineGap=h/3.0)
    if lines is None:
        return 0.0

    min_line = 100
    min_line_pos = 0
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            center_point = [((x1+x2)/2), ((y1+y2)/2)]
            if center_thres == 1 and center_point[1] < 7:
                continue
            if center_point[1] < min_line:
                min_line = center_point[1]
                min_line_pos = i

    angle = 0.0
    cnt = 0
    for x1, y1, x2, y2 in lines[min_line_pos]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30:
            angle += ang
            cnt += 1
    if cnt == 0:
        return 0.0
    return (angle / cnt) * 180 / math.pi

def deskew(src_img, change_cons, center_thres):
    """Deskew image based on computed angle."""
    if change_cons == 1:
        return rotate_image(src_img, compute_skew(change_contrast(src_img), center_thres))
    return rotate_image(src_img, compute_skew(src_img, center_thres))

def iou(box1, box2):
    """Calculate IoU between two bounding boxes [x_center, y_center, w, h]."""
    box1 = [box1[0] - box1[2]/2, box1[1] - box1[3]/2, box1[0] + box1[2]/2, box1[1] + box1[3]/2]
    box2 = [box2[0] - box2[2]/2, box2[1] - box2[3]/2, box2[0] + box2[2]/2, box2[1] + box2[3]/2]
    
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

def linear_equation(x1, y1, x2, y2):
    """Calculate linear equation coefficients."""
    if x2 == x1:
        return None, None
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

def check_point_linear(x, y, x1, y1, x2, y2):
    """Check if point lies on line within tolerance."""
    a, b = linear_equation(x1, y1, x2, y2)
    if a is None:
        return False
    y_pred = a * x + b
    return math.isclose(y_pred, y, abs_tol=3)

def preprocess_frame(frame, input_size=(640, 640)):
    """Preprocess frame for Triton inference."""
    original_height, original_width = frame.shape[:2]
    # Resize frame
    frame_resized = cv2.resize(frame, input_size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    frame_normalized = frame_normalized.transpose((2, 0, 1))
    return frame, frame_resized, frame_normalized, (original_width, original_height)

def postprocess(model_output, original_size, score_threshold=0.35, nms_threshold=0.45):
    """Postprocess Triton inference output."""
    original_width, original_height = original_size
    input_size = 640  # Model input size (640x640)

    # Calculate scaling factors
    scale_x = original_width / input_size
    scale_y = original_height / input_size

    outputs = np.array([cv2.transpose(model_output[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= score_threshold:
            # Original box: [x_center, y_center, w, h]
            x = outputs[0][i][0]
            y = outputs[0][i][1]
            w = outputs[0][i][2]
            h = outputs[0][i][3]

            # Scale bbox to original size
            box = [
                x * scale_x,
                y * scale_y,
                w * scale_x,
                h * scale_y
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold, 0.5)

    num_detections = 0
    output_boxes = []
    output_scores = []
    output_classids = []
    if result_boxes is not None:
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            output_boxes.append(boxes[index])
            output_scores.append(scores[index])
            output_classids.append(class_ids[index])
            num_detections += 1

    num_detections = np.array(num_detections)
    return num_detections, output_boxes, output_scores, output_classids

def triton_infer(client, model_name, frame_normalized):
    """Perform inference using Triton server via gRPC."""
    # Preprocess image for Triton
    # _, _, frame_normalized, _ = preprocess_frame(image)
    
    # Set up inputs
    inputs = []
    input0 = triton_grpc.InferInput("images", frame_normalized[np.newaxis, ...].shape, "FP32")
    input0.set_data_from_numpy(frame_normalized[np.newaxis, ...])
    inputs.append(input0)

    # Set up outputs
    outputs = []
    outputs.append(triton_grpc.InferRequestedOutput("output0"))

    # Query Triton server
    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    return response.as_numpy("output0")

def read_plate(triton_client, plate_crop):
    """Read license plate characters using Triton character recognition model."""
    LP_type = "1"  # Default to single line
    
    # Preprocess plate image for character detection
    _, char_input, char_normalized, char_original_size = preprocess_frame(plate_crop)
    
    # Perform inference with Triton
    char_output = triton_infer(triton_client, "character_detection", char_normalized)
    if char_output is None:
        return "unknown", None
    
    # Postprocess character detection results
    num_detections, boxes, scores, class_ids = postprocess(char_output, char_original_size)
    
    if num_detections < 7 or num_detections > 10:
        return "unknown", (boxes, scores, class_ids)

    # Combine box, label, score
    detections = []
    for i in range(num_detections):
        detections.append({
            "box": boxes[i],
            "score": scores[i],
            "label": int(class_ids[i])
        })

    # Sort by score descending
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

    # Extract centers and labels
    center_list = []
    y_sum = 0
    for det in final_detections:
        x, y, w, h = det["box"]
        x_c = x
        y_c = y
        label = label_to_char(det["label"])
        y_sum += y_c
        center_list.append([x_c, y_c, label])

    if len(center_list) < 7 or len(center_list) > 10:
        return "unknown", (boxes, scores, class_ids)

    # Check for single or double line plate
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

    return license_plate, (boxes, scores, class_ids)

def process_license_plate(triton_client, vehicle_crop):
    """Process vehicle crop to detect and read license plate."""
    processed_plate = preprocess_plate_image(vehicle_crop)

    best_license_plate = "unknown"
    best_char_result = None
    best_conf_score = 0.0

    # Try different rotation configurations
    for cc in range(0, 2):
        for ct in range(0, 2):
            # rotated_plate = deskew(processed_plate, cc, ct)
            rotated_plate = deskew(processed_plate, cc, ct)
            # rotated_plate = preprocess_plate_image(rotated_plate)

            license_plate_text, char_result = read_plate(triton_client, rotated_plate)
            if char_result is not None and len(char_result[1]) > 0:
                total_conf_score = sum(char_result[1]) / len(char_result[1])
            else:
                total_conf_score = 0.0

            if license_plate_text != "unknown" and total_conf_score > best_conf_score:
                best_conf_score = total_conf_score
                best_license_plate = license_plate_text
                best_char_result = char_result

    return best_license_plate, best_char_result, processed_plate

def is_box_inside(inner_box, outer_box):
    """Check if inner_box is fully contained within outer_box."""
    x1_inner, y1_inner, x2_inner, y2_inner = inner_box
    x1_outer, y1_outer, x2_outer, y2_outer = outer_box
    return (x1_inner >= x1_outer and y1_inner >= y1_outer and
            x2_inner <= x2_outer and y2_inner <= y2_outer)