from dataclasses import dataclass
import heapq
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


DEFAULT_IMAGE_PATH = Path("dataset/crack1.jpg")
WEB_OUTPUT_DIR = Path("web_output")
SEGMENT_LENGTH_MM = 10.0
REFERENCE_SQUARE_SIZE_MM = 10.0
REFERENCE_MASK_MARGIN_PX = 12
REFERENCE_DARK_THRESHOLD = 80
REFERENCE_MIN_AREA_RATIO = 0.0002
REFERENCE_MAX_AREA_RATIO = 0.05
REFERENCE_MAX_ASPECT_RATIO = 1.35
REFERENCE_MIN_FILL_RATIO = 0.72
REFERENCE_MAX_MEAN_VALUE = 85.0
MEASUREMENT_EROSION_ITERATIONS = 2
MEASUREMENT_MIN_AREA_RATIO = 0.45
EDGE_SLIVER_MAX_THICKNESS_PX = 1
EDGE_SLIVER_MIN_ASPECT_RATIO = 20.0
EDGE_SLIVER_MAX_AREA = 300
SMALL_COMPONENT_MAX_AREA = 120
MIN_ISOLATED_COMPONENT_LENGTH_MM = 4.0
DARK_VALUE_THRESHOLD = 145
ENABLE_DYNAMIC_DARK_THRESHOLD = False
DYNAMIC_STRONG_DARK_MIN = 125
DYNAMIC_STRONG_DARK_MAX = 155
DYNAMIC_STRONG_DARK_OFFSET = 10
DYNAMIC_BRIGHT_MEDIAN_THRESHOLD = 195
DYNAMIC_HIGH_MEDIAN_THRESHOLD = 210
DYNAMIC_MIN_STD_FOR_BOOST = 22
DYNAMIC_HIGH_STD_FOR_BOOST = 25
DYNAMIC_MEDIUM_THRESHOLD_BOOST = 5
DYNAMIC_HIGH_THRESHOLD_BOOST = 10
DYNAMIC_WEAK_DARK_MIN = 150
DYNAMIC_WEAK_DARK_MAX = 185
DYNAMIC_WEAK_DARK_OFFSET = 22
DARK_WEAK_MAX_THRESHOLD = 220
DARK_WEAK_MIN_THRESHOLD = 175
DARK_WEAK_BACKGROUND_DELTA = 12
BLACKHAT_KERNEL = 19
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 7
MIN_COMPONENT_AREA = 40
MIN_COMPONENT_ASPECT_RATIO = 2.5
LARGE_COMPONENT_MIN_ASPECT_RATIO = 1.35
NEARBY_COMPONENT_KERNEL = 45
GAP_BRIDGE_KERNEL = 7
MIN_PATH_LENGTH_PX = 2
LABEL_EVERY_N_SEGMENTS = 5
LENGTH_SMOOTHING_EPSILON_RATIO = 0.01
MIN_SIGNIFICANT_BRANCH_LENGTH_MM = 3.0
MIN_SIGNIFICANT_BRANCH_RATIO = 0.05
BRANCHED_COMPONENT_MIN_LENGTH_MM = 2.5
BRANCHED_COMPONENT_BRANCH_RATIO = 0.18
MIN_COMPONENT_PATH_COVERAGE_RATIO = 0.75
RELAXED_MAX_PATH_Q75_INTENSITY = 175.0
CONNECTOR_MAX_PATH_Q75_INTENSITY = 185.0
MAX_PATH_Q75_INTENSITY = 160.0
AUTO_PRIMARY_MODE_PATH_COUNT = 20
PRIMARY_CENTER_BAND_RATIO = 0.18
FORCE_PRIMARY_CENTER_CRACK = False
LARGE_COMPONENT_MAX_FILL_RATIO = 0.22
PRIMARY_MODE_MIN_HEIGHT_RATIO = 0.55
BLACKHAT_STD_THRESHOLD_RATIO = 0.8
BLACKHAT_MIN_THRESHOLD = 20
BLACKHAT_WEAK_THRESHOLD_RATIO = 0.55
MAX_COMPONENT_Q75_INTENSITY = 135.0
MAX_RESCUED_COMPONENT_Q75_INTENSITY = 235.0
MIN_RESCUED_COMPONENT_PCA_RATIO = 12.0
MIN_RESCUED_COMPONENT_LENGTH_PX = 15.0
MAX_RESCUED_COMPONENT_FILL_RATIO = 0.35
MIN_RESCUED_DARK_PIXEL_RATIO = 0.015
MIN_RESCUED_BLACKHAT_MEAN = 18.0
ENABLE_FAINT_BORDER_RESCUE = True
AUTO_FAINT_BORDER_RESCUE = False
AUTO_RESCUE_MAX_LENGTH_INCREASE_RATIO = 0.25
AUTO_RESCUE_MAX_SEGMENT_INCREASE_RATIO = 0.40
AUTO_RESCUE_MIN_ADDED_LENGTH_MM = 5.0
MIN_COMPONENT_PCA_RATIO = 12.0
MIN_COMPONENT_FILTER_PATH_LENGTH_PX = 5.0
MIN_BRANCH_WIDTH_REFERENCE_RATIO = 0.04
ENABLE_BINARY_THIN_BRANCH_PRUNING = True
MIN_THIN_BRANCH_PRUNE_LENGTH_PX = 8.0
THIN_BRANCH_PRUNE_PASSES = 3
MAX_WIDE_DARK_COMPONENT_FILL_RATIO = 0.35
MIN_WIDE_DARK_COMPONENT_LENGTH_RATIO = 4.0
MIN_VALID_IMAGE_SIDE_PX = 100
MIN_LIGHT_BACKGROUND_MEDIAN = 115.0
MAX_DARK_BACKGROUND_RATIO = 0.30
MAX_EXTREME_EXPOSURE_RATIO = 0.35
MIN_LAPLACIAN_VARIANCE = 0.0


@dataclass
class SegmentMeasurement:
    index: int
    branch_id: int
    length_px: float
    length_mm: float
    mean_width_px: float
    mean_width_mm: float
    max_width_px: float
    max_width_mm: float
    points: list
    passes_width: bool = True


@dataclass
class SquareCalibration:
    pixels_per_mm: float
    reference_side_px: float
    reference_corners: np.ndarray
    reference_debug_image: np.ndarray


@dataclass
class ToneProfile:
    background_median: float
    background_p10: float
    background_p25: float
    background_p75: float
    background_p90: float
    background_std: float
    strong_dark_threshold: int
    weak_dark_threshold: int


def build_reference_exclusion_mask(shape, calibration: SquareCalibration, margin_px: int = REFERENCE_MASK_MARGIN_PX):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, calibration.reference_corners.astype(np.int32), 255)
    kernel_size = margin_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def clamp_int(value, lower, upper):
    return int(max(lower, min(upper, round(float(value)))))


# Cac hang so khong gian ben duoi duoc tinh chinh o do phan giai tham chieu
# REFERENCE_DESIGN_PIXELS_PER_MM. Vi anh dau vao khong co dinh, moi tham so
# khong gian duoc co gian theo ti le pixels_per_mm thuc / ppm tham chieu:
#  - chieu dai/kernel: co gian tuyen tinh
#  - dien tich: co gian theo binh phuong
# pixels_per_mm luon suy tu o vuong den 10mm (calibration.reference_side_px),
# nho vay thuat toan tu dong thich nghi voi anh phan giai bat ky.
REFERENCE_DESIGN_PIXELS_PER_MM = 1.35


def spatial_scale(pixels_per_mm) -> float:
    if not pixels_per_mm or pixels_per_mm <= 0:
        return 1.0
    return float(pixels_per_mm) / REFERENCE_DESIGN_PIXELS_PER_MM


def scale_len(value_px: float, pixels_per_mm) -> float:
    return float(value_px) * spatial_scale(pixels_per_mm)


def scale_area(value_px: float, pixels_per_mm) -> float:
    factor = spatial_scale(pixels_per_mm)
    return float(value_px) * factor * factor


def scale_odd_kernel(value_px: float, pixels_per_mm, min_size: int = 3) -> int:
    size = int(round(float(value_px) * spatial_scale(pixels_per_mm)))
    size = max(min_size, size)
    if size % 2 == 0:
        size += 1
    return size


def scale_px(mm: float, pixels_per_mm) -> int:
    return int(max(1, round(float(mm) * float(pixels_per_mm))))


# Khe ho duoc bac cau theo do phan giai, nhung khong bao gio noi khe vat ly
# lon hon GAP_BRIDGE_MAX_MM (tranh dinh nham 2 vet nut khac nhau).
GAP_BRIDGE_MAX_MM = 1.2
# Gai/nhanh cut gia ngan hon nguong nay (theo mm) bi cat truoc khi trich path,
# giup duong ve chinh xac hon ma khong lam dut duong chinh.
SPUR_PRUNE_MM = 0.5
# Noi 2 dau mut skeleton khi: khoang cach <= ENDPOINT_BRIDGE_MAX_MM va huong
# tiep tuyen 2 dau gan thang hang (lech goc <= ENDPOINT_BRIDGE_MAX_ANGLE_DEG).
ENDPOINT_BRIDGE_MAX_MM = 1.5
ENDPOINT_BRIDGE_MAX_ANGLE_DEG = 35.0


def dynamic_bridge_kernel_size(pixels_per_mm) -> int:
    size = scale_odd_kernel(GAP_BRIDGE_KERNEL, pixels_per_mm, min_size=3)
    if pixels_per_mm and pixels_per_mm > 0:
        cap = scale_px(GAP_BRIDGE_MAX_MM, pixels_per_mm)
        if cap % 2 == 0:
            cap += 1
        size = min(size, max(3, cap))
    return size


def compute_tone_profile(
    gray_image: np.ndarray,
    calibration: SquareCalibration | None = None,
    use_dynamic_thresholds=ENABLE_DYNAMIC_DARK_THRESHOLD,
) -> ToneProfile:
    if calibration is not None:
        exclusion_mask = build_reference_exclusion_mask(gray_image.shape, calibration)
        values = gray_image[exclusion_mask == 0]
    else:
        values = gray_image.reshape(-1)

    if values.size == 0:
        values = gray_image.reshape(-1)

    background_median = float(np.median(values))
    background_p10 = float(np.percentile(values, 10))
    background_p25 = float(np.percentile(values, 25))
    background_p75 = float(np.percentile(values, 75))
    background_p90 = float(np.percentile(values, 90))
    background_std = float(np.std(values))

    if use_dynamic_thresholds:
        base_strong_dark_threshold = clamp_int(
            background_p10 - DYNAMIC_STRONG_DARK_OFFSET,
            DYNAMIC_STRONG_DARK_MIN,
            DARK_VALUE_THRESHOLD,
        )
        threshold_boost = 0
        if background_median >= DYNAMIC_HIGH_MEDIAN_THRESHOLD and background_std >= DYNAMIC_HIGH_STD_FOR_BOOST:
            threshold_boost = DYNAMIC_HIGH_THRESHOLD_BOOST
        elif background_median >= DYNAMIC_BRIGHT_MEDIAN_THRESHOLD and background_std >= DYNAMIC_MIN_STD_FOR_BOOST:
            threshold_boost = DYNAMIC_MEDIUM_THRESHOLD_BOOST

        strong_dark_threshold = clamp_int(
            base_strong_dark_threshold + threshold_boost,
            DYNAMIC_STRONG_DARK_MIN,
            DYNAMIC_STRONG_DARK_MAX,
        )
        weak_dark_threshold = clamp_int(
            background_p25 - DYNAMIC_WEAK_DARK_OFFSET,
            DYNAMIC_WEAK_DARK_MIN,
            DYNAMIC_WEAK_DARK_MAX,
        )
        weak_dark_threshold = max(weak_dark_threshold, strong_dark_threshold)
    else:
        strong_dark_threshold = DARK_VALUE_THRESHOLD
        weak_dark_threshold = DARK_VALUE_THRESHOLD

    return ToneProfile(
        background_median=background_median,
        background_p10=background_p10,
        background_p25=background_p25,
        background_p75=background_p75,
        background_p90=background_p90,
        background_std=background_std,
        strong_dark_threshold=strong_dark_threshold,
        weak_dark_threshold=weak_dark_threshold,
    )


def tone_profile_to_payload(profile: ToneProfile) -> dict:
    return {
        "background_median": round(profile.background_median, 3),
        "background_p10": round(profile.background_p10, 3),
        "background_p25": round(profile.background_p25, 3),
        "background_p75": round(profile.background_p75, 3),
        "background_p90": round(profile.background_p90, 3),
        "background_std": round(profile.background_std, 3),
        "strong_dark_threshold": int(profile.strong_dark_threshold),
        "weak_dark_threshold": int(profile.weak_dark_threshold),
    }


def validate_input_image(image: np.ndarray, calibration: SquareCalibration) -> dict:
    if image is None or image.size == 0:
        raise ValueError("Anh dau vao rong hoac khong doc duoc.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Anh dau vao phai la anh mau BGR/RGB hop le.")

    image_height, image_width = image.shape[:2]
    if min(image_height, image_width) < MIN_VALID_IMAGE_SIDE_PX:
        raise ValueError(
            f"Anh qua nho. Can moi chieu >= {MIN_VALID_IMAGE_SIDE_PX}px de detect vet nut va o chuan."
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    exclusion_mask = build_reference_exclusion_mask(gray.shape, calibration)
    background_mask = exclusion_mask == 0
    background_values = gray[background_mask]
    if background_values.size < image_height * image_width * 0.5:
        raise ValueError("O chuan chiem qua nhieu anh, khong du vung nen tuong de validate.")

    background_median = float(np.median(background_values))
    dark_background_ratio = float(np.mean(background_values < 70))
    under_exposed_ratio = float(np.mean(background_values < 35))
    over_exposed_ratio = float(np.mean(background_values > 245))
    extreme_exposure_ratio = under_exposed_ratio + over_exposed_ratio
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if background_median < MIN_LIGHT_BACKGROUND_MEDIAN:
        raise ValueError(
            "Nen tuong khong thuoc nhom mau sang theo dac ta "
            "(trang, xam trang, vang nhat) hoac anh dang thieu sang."
        )
    if dark_background_ratio > MAX_DARK_BACKGROUND_RATIO:
        raise ValueError(
            "Anh co qua nhieu vung toi/hoa van den, de nham voi vet nut theo dac ta dau vao."
        )
    if extreme_exposure_ratio > MAX_EXTREME_EXPOSURE_RATIO:
        raise ValueError("Anh bi thieu sang/chay sang qua muc, khong dat dieu kien dau vao.")
    if laplacian_variance < MIN_LAPLACIAN_VARIANCE:
        raise ValueError("Anh bi mo/rung, khong ro canh vet nut hoac canh o chuan.")

    return {
        "background_median": round(background_median, 3),
        "dark_background_ratio": round(dark_background_ratio, 6),
        "extreme_exposure_ratio": round(extreme_exposure_ratio, 6),
        "laplacian_variance": round(laplacian_variance, 3),
        "reference_square_size_mm": REFERENCE_SQUARE_SIZE_MM,
        "passed": True,
    }


def build_degree_map(skeleton: np.ndarray) -> dict:
    points = np.column_stack(np.where(skeleton > 0))
    if len(points) == 0:
        return {}

    graph = build_graph(points)
    return {node: len(neighbors) for node, neighbors in graph.items()}


def prune_skeleton_spurs(skeleton: np.ndarray, pixels_per_mm, max_passes: int = 3) -> np.ndarray:
    """Cat cac gai/nhanh cut gia ngan moc ra tu mot nut giao. Khong cat doan
    la mot fragment doc lap (ca 2 dau deu la endpoint) -> khong lam dut duong chinh."""
    cleaned = skeleton.copy()
    threshold_px = max(3.0, float(scale_px(SPUR_PRUNE_MM, pixels_per_mm)))

    for _ in range(max_passes):
        points = np.column_stack(np.where(cleaned > 0))
        if len(points) == 0:
            break
        graph = build_graph(points)
        if not graph:
            break
        endpoints = [node for node, nb in graph.items() if len(nb) == 1]
        to_remove = []
        for endpoint in endpoints:
            branch = [endpoint]
            previous = None
            current = endpoint
            while True:
                neighbors = [n for n, _ in graph.get(current, []) if n != previous]
                if len(graph.get(current, [])) == 1 and current != endpoint:
                    far_degree = 1
                    break
                if not neighbors:
                    far_degree = len(graph.get(current, []))
                    break
                nxt = neighbors[0]
                if len(graph.get(nxt, [])) != 2:
                    far_degree = len(graph.get(nxt, []))
                    break
                branch.append(nxt)
                previous, current = current, nxt
                if compute_path_length(branch) > threshold_px:
                    far_degree = 2
                    break
            if far_degree >= 3 and compute_path_length(branch) < threshold_px:
                to_remove.extend(branch)
        if not to_remove:
            break
        for y, x in to_remove:
            cleaned[y, x] = 0
    return cleaned


def _endpoint_tangent(node, graph, steps):
    previous = None
    current = node
    walked = [node]
    for _ in range(steps):
        neighbors = [n for n, _ in graph.get(current, []) if n != previous]
        if not neighbors:
            break
        nxt = neighbors[0]
        walked.append(nxt)
        previous, current = current, nxt
    inward = walked[-1]
    vec = np.array([node[0] - inward[0], node[1] - inward[1]], dtype=np.float32)
    norm = float(np.hypot(vec[0], vec[1]))
    if norm < 1e-6:
        return None
    return vec / norm


def bridge_skeleton_endpoints(skeleton: np.ndarray, pixels_per_mm) -> np.ndarray:
    points = np.column_stack(np.where(skeleton > 0))
    if len(points) == 0:
        return skeleton
    graph = build_graph(points)
    endpoints = [node for node, nb in graph.items() if len(nb) == 1]
    if len(endpoints) < 2:
        return skeleton

    max_gap = max(3.0, float(scale_px(ENDPOINT_BRIDGE_MAX_MM, pixels_per_mm)))
    max_angle = np.deg2rad(ENDPOINT_BRIDGE_MAX_ANGLE_DEG)
    tangent_steps = max(3, scale_px(0.6, pixels_per_mm))
    tangents = {e: _endpoint_tangent(e, graph, tangent_steps) for e in endpoints}

    bridged = skeleton.copy()
    used = set()
    candidates = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            e1, e2 = endpoints[i], endpoints[j]
            dist = float(np.hypot(e2[0] - e1[0], e2[1] - e1[1]))
            if dist <= 1.0 or dist > max_gap:
                continue
            t1, t2 = tangents.get(e1), tangents.get(e2)
            if t1 is None or t2 is None:
                continue
            gap = np.array([e2[0] - e1[0], e2[1] - e1[1]], dtype=np.float32) / dist
            ang1 = np.arccos(np.clip(float(np.dot(t1, gap)), -1.0, 1.0))
            ang2 = np.arccos(np.clip(float(np.dot(t2, -gap)), -1.0, 1.0))
            if ang1 <= max_angle and ang2 <= max_angle:
                candidates.append((dist, e1, e2))

    for dist, e1, e2 in sorted(candidates, key=lambda c: c[0]):
        if e1 in used or e2 in used:
            continue
        cv2.line(bridged, (e1[1], e1[0]), (e2[1], e2[0]), 255, 1, cv2.LINE_8)
        used.add(e1)
        used.add(e2)
    return bridged


def extract_component_centerlines(binary: np.ndarray, pixels_per_mm: float, gray_image=None):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    paths = []
    image_height, image_width = binary.shape[:2]

    min_component_area = scale_area(MIN_COMPONENT_AREA, pixels_per_mm)
    min_path_length_px = scale_len(MIN_PATH_LENGTH_PX, pixels_per_mm)
    edge_sliver_max_thickness = scale_len(EDGE_SLIVER_MAX_THICKNESS_PX, pixels_per_mm)
    edge_sliver_max_area = scale_area(EDGE_SLIVER_MAX_AREA, pixels_per_mm)
    small_component_max_area = scale_area(SMALL_COMPONENT_MAX_AREA, pixels_per_mm)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        short_side = max(min(width, height), 1)
        aspect_ratio = max(width, height) / short_side
        touches_border = (
            x == 0
            or y == 0
            or (x + width) >= image_width
            or (y + height) >= image_height
        )

        component = np.zeros_like(binary)
        component[labels == label] = 255
        component_skeleton = skeletonize(component)
        component_skeleton = prune_skeleton_spurs(component_skeleton, pixels_per_mm)
        component_skeleton = bridge_skeleton_endpoints(component_skeleton, pixels_per_mm)
        component_paths = keep_significant_component_paths(
            component_skeleton,
            pixels_per_mm,
            gray_image=gray_image,
        )
        if not component_paths:
            continue

        smoothed_length_px = max(compute_smoothed_path_length(path) for path in component_paths)
        if smoothed_length_px < min_path_length_px:
            continue

        if (
            touches_border
            and short_side <= edge_sliver_max_thickness
            and aspect_ratio >= EDGE_SLIVER_MIN_ASPECT_RATIO
            and area <= edge_sliver_max_area
        ):
            continue

        if (
            area <= small_component_max_area
            and smoothed_length_px < MIN_ISOLATED_COMPONENT_LENGTH_MM * pixels_per_mm
        ):
            continue

        paths.extend(component_paths)

    paths.sort(key=compute_path_length, reverse=True)
    return paths


def order_quad_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    point_sums = points.sum(axis=1)
    point_diffs = np.diff(points, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(point_sums)]
    ordered[2] = points[np.argmax(point_sums)]
    ordered[1] = points[np.argmin(point_diffs)]
    ordered[3] = points[np.argmax(point_diffs)]
    return ordered


def build_reference_square_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, fixed_mask = cv2.threshold(gray, REFERENCE_DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    square_mask = cv2.bitwise_and(fixed_mask, otsu_mask)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    square_mask = cv2.morphologyEx(square_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    square_mask = cv2.morphologyEx(square_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return square_mask


def detect_reference_square(image: np.ndarray):
    image_height, image_width = image.shape[:2]
    image_area = float(image_height * image_width)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    best_candidate = None
    square_mask = build_reference_square_mask(image)
    contours, _ = cv2.findContours(square_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < 25:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) < 4 or len(approx) > 6:
            continue
        if not cv2.isContourConvex(approx):
            continue

        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width <= 1.0 or height <= 1.0:
            continue

        aspect_ratio = max(width, height) / max(min(width, height), 1e-6)
        rectangle_area = float(width * height)
        area_ratio = rectangle_area / image_area
        if area_ratio < REFERENCE_MIN_AREA_RATIO or area_ratio > REFERENCE_MAX_AREA_RATIO:
            continue
        if aspect_ratio > REFERENCE_MAX_ASPECT_RATIO:
            continue

        reference_corners = cv2.boxPoints(rect).astype(np.float32)
        reference_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.fillConvexPoly(reference_mask, reference_corners.astype(np.int32), 255)

        fill_ratio = float(contour_area) / max(rectangle_area, 1.0)
        mean_value = float(cv2.mean(gray, mask=reference_mask)[0])
        if fill_ratio < REFERENCE_MIN_FILL_RATIO:
            continue
        if mean_value > REFERENCE_MAX_MEAN_VALUE:
            continue

        reference_side_px = float((width + height) * 0.5)
        pixels_per_mm = reference_side_px / REFERENCE_SQUARE_SIZE_MM
        squareness_score = 1.0 / (1.0 + abs(1.0 - aspect_ratio))
        candidate_score = fill_ratio * squareness_score * reference_side_px

        debug_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_image, [reference_corners.astype(np.int32)], -1, (0, 255, 255), 2, cv2.LINE_AA)
        center = tuple(np.mean(reference_corners, axis=0).astype(int))
        cv2.circle(debug_image, center, 3, (0, 255, 0), -1)

        calibration = SquareCalibration(
            pixels_per_mm=float(pixels_per_mm),
            reference_side_px=reference_side_px,
            reference_corners=reference_corners,
            reference_debug_image=debug_image,
        )

        if best_candidate is None or candidate_score > best_candidate[0]:
            best_candidate = (candidate_score, calibration)

    if best_candidate is None:
        return None

    return best_candidate[1]


def skeletonize(binary: np.ndarray) -> np.ndarray:
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary)

    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = binary.copy()

    while True:
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        residue = cv2.subtract(work, opened)
        skel = cv2.bitwise_or(skel, residue)
        work = eroded

        if cv2.countNonZero(work) == 0:
            break

    return cv2.bitwise_and(skel, binary)


def build_graph(points: np.ndarray):
    point_set = {tuple(point) for point in points.tolist()}
    graph = {}

    for y, x in point_set:
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in point_set:
                    if dy != 0 and dx != 0:
                        if (y, nx) in point_set or (ny, x) in point_set:
                            continue
                    neighbors.append(((ny, nx), float(np.hypot(dx, dy))))
        graph[(y, x)] = neighbors

    return graph


def edge_key(a, b):
    return tuple(sorted((a, b)))


def dijkstra_graph(graph, start):
    distances = {start: 0.0}
    previous = {}
    heap = [(0.0, start)]

    while heap:
        current_distance, node = heapq.heappop(heap)
        if current_distance > distances.get(node, float("inf")):
            continue

        for neighbor, weight in graph[node]:
            new_distance = current_distance + weight
            if new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                previous[neighbor] = node
                heapq.heappush(heap, (new_distance, neighbor))

    return distances, previous


def reconstruct_graph_path(previous, start, end):
    path = [end]
    while path[-1] != start:
        parent = previous.get(path[-1])
        if parent is None:
            return []
        path.append(parent)
    path.reverse()
    return path


def compute_path_length(path):
    if len(path) < 2:
        return 0.0

    total = 0.0
    for idx in range(1, len(path)):
        y1, x1 = path[idx - 1]
        y2, x2 = path[idx]
        total += float(np.hypot(x2 - x1, y2 - y1))
    return total


def compute_smoothed_path_length(path):
    if len(path) < 2:
        return 0.0

    polyline = np.array([(float(x), float(y)) for y, x in path], dtype=np.float32).reshape(-1, 1, 2)
    epsilon = max(1.0, LENGTH_SMOOTHING_EPSILON_RATIO * cv2.arcLength(polyline, False))
    simplified = cv2.approxPolyDP(polyline, epsilon, False)
    return float(cv2.arcLength(simplified, False))


def compute_total_smoothed_length(paths):
    return float(sum(compute_smoothed_path_length(path) for path in paths))


def compute_path_intensity_q75(path, gray_image):
    if gray_image is None or len(path) == 0:
        return 0.0

    samples = np.array([gray_image[y, x] for y, x in path], dtype=np.float32)
    return float(np.percentile(samples, 75))


def compute_dynamic_weak_dark_threshold(gray_image: np.ndarray, reference_corners=None) -> int:
    mask = np.ones(gray_image.shape, dtype=np.uint8) * 255
    if reference_corners is not None:
        reference_mask = np.zeros_like(gray_image, dtype=np.uint8)
        cv2.fillConvexPoly(reference_mask, reference_corners.astype(np.int32), 255)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (REFERENCE_MASK_MARGIN_PX * 2 + 1, REFERENCE_MASK_MARGIN_PX * 2 + 1),
        )
        reference_mask = cv2.dilate(reference_mask, kernel, iterations=1)
        mask[reference_mask > 0] = 0

    background_values = gray_image[mask > 0]
    if background_values.size == 0:
        background_values = gray_image.reshape(-1)

    background_median = float(np.median(background_values))
    weak_threshold = int(background_median - DARK_WEAK_BACKGROUND_DELTA)
    weak_threshold = max(DARK_WEAK_MIN_THRESHOLD, weak_threshold)
    weak_threshold = min(DARK_WEAK_MAX_THRESHOLD, weak_threshold)
    weak_threshold = max(DARK_VALUE_THRESHOLD, weak_threshold)
    return int(weak_threshold)


def keep_hysteresis_candidates(strong_mask: np.ndarray, weak_mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weak_mask, connectivity=8)
    if num_labels <= 1:
        return weak_mask

    strong_labels = set(np.unique(labels[strong_mask > 0]).astype(int).tolist())
    strong_labels.discard(0)

    kept = np.zeros_like(weak_mask)
    for label in range(1, num_labels):
        if label in strong_labels:
            kept[labels == label] = 255

    return kept


def compute_component_pca_ratio(component_mask: np.ndarray) -> float:
    points = np.column_stack(np.where(component_mask > 0))
    if len(points) < 3:
        return 999.0

    points = points.astype(np.float32)
    points -= points.mean(axis=0, keepdims=True)
    covariance = np.cov(points.T)
    eigenvalues, _ = np.linalg.eigh(covariance)
    eigenvalues = np.sort(eigenvalues)[::-1]
    if eigenvalues[1] <= 1e-6:
        return 999.0

    return float(eigenvalues[0] / eigenvalues[1])


def compute_path_coverage(paths, skeleton: np.ndarray) -> float:
    total_points = int(cv2.countNonZero(skeleton))
    if total_points <= 0:
        return 0.0

    covered_points = len(collect_unique_path_points(paths))
    return covered_points / float(total_points)


def path_endpoint_is_near_selected(endpoint, selected_points, radius=1):
    y, x = endpoint
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if (y + dy, x + dx) in selected_points:
                return True
    return False


def path_connects_selected_paths(path, selected_paths) -> bool:
    if not path or not selected_paths:
        return False

    selected_points = {tuple(point) for selected_path in selected_paths for point in selected_path}
    return (
        path_endpoint_is_near_selected(tuple(path[0]), selected_points)
        and path_endpoint_is_near_selected(tuple(path[-1]), selected_points)
    )


def keep_significant_component_paths(skeleton: np.ndarray, pixels_per_mm: float, gray_image=None):
    component_paths = [
        path
        for path in extract_all_paths(skeleton)
        if len(path) > 1
    ]

    if not component_paths:
        fallback_path = longest_centerline_path(skeleton)
        return [fallback_path] if len(fallback_path) > 1 else []

    skeleton_points = np.column_stack(np.where(skeleton > 0))
    graph = build_graph(skeleton_points) if len(skeleton_points) > 0 else {}
    has_branching = any(len(neighbors) >= 3 for neighbors in graph.values())

    measured_paths = []
    for path in component_paths:
        length_px = compute_smoothed_path_length(path)
        if length_px < MIN_PATH_LENGTH_PX:
            continue
        q75_intensity = compute_path_intensity_q75(path, gray_image) if gray_image is not None else 0.0
        measured_paths.append((path, length_px, q75_intensity))
    if not measured_paths:
        return []

    longest_length_px = max(length_px for _, length_px, _ in measured_paths)
    relative_ratio = MIN_SIGNIFICANT_BRANCH_RATIO
    absolute_length_mm = MIN_SIGNIFICANT_BRANCH_LENGTH_MM
    if has_branching:
        relative_ratio = min(relative_ratio, BRANCHED_COMPONENT_BRANCH_RATIO)
        absolute_length_mm = min(absolute_length_mm, BRANCHED_COMPONENT_MIN_LENGTH_MM)

    length_threshold_px = max(
        MIN_PATH_LENGTH_PX,
        absolute_length_mm * pixels_per_mm,
    )
    if len(measured_paths) > 1:
        length_threshold_px = max(
            length_threshold_px,
            longest_length_px * relative_ratio,
        )

    significant_paths = [
        path
        for path, length_px, _ in measured_paths
        if length_px >= length_threshold_px
    ]

    if significant_paths and gray_image is not None:
        dark_paths = [
            path
            for path, _, q75_intensity in measured_paths
            if id(path) in {id(selected_path) for selected_path in significant_paths}
            and q75_intensity <= MAX_PATH_Q75_INTENSITY
        ]
        if dark_paths:
            significant_paths = dark_paths

    if significant_paths:
        coverage_ratio = compute_path_coverage(significant_paths, skeleton)
        if coverage_ratio < MIN_COMPONENT_PATH_COVERAGE_RATIO:
            selected_ids = {id(path) for path in significant_paths}
            remaining_paths = sorted(
                [
                    (path, length_px, q75_intensity)
                    for path, length_px, q75_intensity in measured_paths
                    if id(path) not in selected_ids
                ],
                key=lambda item: (item[2] <= RELAXED_MAX_PATH_Q75_INTENSITY, item[1]),
                reverse=True,
            )
            for path, _, q75_intensity in remaining_paths:
                if gray_image is not None and q75_intensity > RELAXED_MAX_PATH_Q75_INTENSITY:
                    is_connector = (
                        q75_intensity <= CONNECTOR_MAX_PATH_Q75_INTENSITY
                        and path_connects_selected_paths(path, significant_paths)
                    )
                    if not is_connector:
                        continue
                significant_paths.append(path)
                coverage_ratio = compute_path_coverage(significant_paths, skeleton)
                if coverage_ratio >= MIN_COMPONENT_PATH_COVERAGE_RATIO:
                    break

        significant_paths.sort(key=compute_path_length, reverse=True)
        return significant_paths

    fallback_path = max(measured_paths, key=lambda item: item[1])[0]
    return [fallback_path]


def extract_all_paths(skeleton: np.ndarray):
    points = np.column_stack(np.where(skeleton > 0))
    if len(points) == 0:
        return []

    graph = build_graph(points)
    special_nodes = [node for node, neighbors in graph.items() if len(neighbors) != 2]
    visited_edges = set()
    paths = []

    def walk_path(start, next_node):
        path = [start, next_node]
        visited_edges.add(edge_key(start, next_node))
        previous = start
        current = next_node

        while len(graph[current]) == 2:
            candidates = [neighbor for neighbor, _ in graph[current] if neighbor != previous]
            if not candidates:
                break

            nxt = candidates[0]
            key = edge_key(current, nxt)
            if key in visited_edges:
                break

            path.append(nxt)
            visited_edges.add(key)
            previous, current = current, nxt

        return path

    for start in special_nodes:
        for neighbor, _ in graph[start]:
            key = edge_key(start, neighbor)
            if key in visited_edges:
                continue
            path = walk_path(start, neighbor)
            if len(path) > 1:
                paths.append(path)

    for start, neighbors in graph.items():
        for neighbor, _ in neighbors:
            key = edge_key(start, neighbor)
            if key in visited_edges:
                continue
            path = walk_path(start, neighbor)
            if len(path) > 1:
                paths.append(path)

    paths.sort(key=compute_path_length, reverse=True)
    return [path for path in paths if compute_path_length(path) >= MIN_PATH_LENGTH_PX]


def longest_centerline_path(skeleton: np.ndarray):
    points = np.column_stack(np.where(skeleton > 0))
    if len(points) == 0:
        return []

    graph = build_graph(points)
    graph = {node: neighbors for node, neighbors in graph.items() if neighbors}
    if not graph:
        return []

    endpoints = [node for node, neighbors in graph.items() if len(neighbors) == 1]
    start_nodes = endpoints if endpoints else [next(iter(graph))]

    best_length = -1.0
    best_path = []

    for start in start_nodes:
        distances, previous = dijkstra_graph(graph, start)
        if not distances:
            continue

        end = max(distances, key=distances.get)
        path_length = distances[end]
        if path_length <= best_length:
            continue

        path = reconstruct_graph_path(previous, start, end)

        if len(path) > 1:
            best_length = path_length
            best_path = path

    return best_path


def preprocess(
    image: np.ndarray,
    reference_corners=None,
    faint_border_rescue=False,
    tone_profile: ToneProfile | None = None,
    min_keep_width_px=None,
    pixels_per_mm=None,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if tone_profile is None:
        tone_profile = compute_tone_profile(gray)

    # Pad anh truoc khi xu ly de tranh border effect cua morphological ops
    PAD = 32
    gray_p = cv2.copyMakeBorder(gray, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT)

    blur = cv2.GaussianBlur(gray_p, (5, 5), 0)
    # CLAHE thich nghi theo do tuong phan nen: nen do lech chuan thap (anh
    # bet, tuong phan yeu) -> clip cao de boost manh; nen do lech chuan cao
    # (anh net) -> clip thap de tranh khuech dai nhieu.
    clahe_clip = float(np.clip(2.5 + (28.0 - tone_profile.background_std) * 0.06, 1.5, 4.0))
    clahe_p = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(blur)

    blackhat_kernel_size = scale_odd_kernel(BLACKHAT_KERNEL, pixels_per_mm, min_size=3)
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (blackhat_kernel_size, blackhat_kernel_size)
    )
    blackhat_p = cv2.morphologyEx(clahe_p, cv2.MORPH_BLACKHAT, blackhat_kernel)
    blackhat_p = cv2.GaussianBlur(blackhat_p, (5, 5), 0)

    # Adaptive threshold cung chay tren anh padded de tranh border effect
    adaptive_block_size = scale_odd_kernel(ADAPTIVE_BLOCK_SIZE, pixels_per_mm, min_size=3)
    adaptive_mask_p = cv2.adaptiveThreshold(
        clahe_p,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        ADAPTIVE_C,
    )

    # Crop tat ca ve kich thuoc goc sau khi xu ly tren anh padded
    clahe        = clahe_p       [PAD:-PAD, PAD:-PAD]
    blackhat     = blackhat_p    [PAD:-PAD, PAD:-PAD]
    adaptive_mask = adaptive_mask_p[PAD:-PAD, PAD:-PAD]

    blackhat_threshold = max(
        BLACKHAT_MIN_THRESHOLD,
        int(float(np.mean(blackhat) + np.std(blackhat) * BLACKHAT_STD_THRESHOLD_RATIO)),
    )
    _, blackhat_mask = cv2.threshold(blackhat, blackhat_threshold, 255, cv2.THRESH_BINARY)
    dark_threshold = tone_profile.weak_dark_threshold if faint_border_rescue else tone_profile.strong_dark_threshold
    _, dark_mask = cv2.threshold(clahe, dark_threshold, 255, cv2.THRESH_BINARY_INV)

    line_candidate_mask = cv2.bitwise_and(blackhat_mask, adaptive_mask)
    binary = cv2.bitwise_and(line_candidate_mask, dark_mask)

    bridge_size = dynamic_bridge_kernel_size(pixels_per_mm)
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_size, bridge_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, bridge_kernel, iterations=2)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
    if reference_corners is not None:
        reference_mask = np.zeros_like(binary)
        cv2.fillConvexPoly(reference_mask, reference_corners.astype(np.int32), 255)
        mask_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (REFERENCE_MASK_MARGIN_PX * 2 + 1, REFERENCE_MASK_MARGIN_PX * 2 + 1),
        )
        reference_mask = cv2.dilate(reference_mask, mask_kernel, iterations=1)
        binary[reference_mask > 0] = 0

    # Mask vung den tuyet doi (pixel = 0) do undistort tao ra
    # Anh that khong co pixel = 0, nen day chi loai artifact cua undistort
    black_border = (gray < 10).astype(np.uint8) * 255
    if cv2.countNonZero(black_border) > 0:
        expand = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        black_border = cv2.dilate(black_border, expand, iterations=2)
        binary[black_border > 0] = 0

    return keep_all_crack_components(
        binary,
        gray_image=gray,
        blackhat_image=blackhat,
        faint_border_rescue=faint_border_rescue,
        dark_threshold=dark_threshold,
        min_keep_width_px=min_keep_width_px,
        pixels_per_mm=pixels_per_mm,
    )


def build_measurement_binary(binary: np.ndarray) -> np.ndarray:
    if MEASUREMENT_EROSION_ITERATIONS <= 0:
        return binary.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(binary, kernel, iterations=MEASUREMENT_EROSION_ITERATIONS)
    if cv2.countNonZero(eroded) < cv2.countNonZero(binary) * MEASUREMENT_MIN_AREA_RATIO:
        return binary.copy()

    return eroded


def prune_thin_crack_pixels(binary: np.ndarray, calibration: SquareCalibration):
    min_width_px = float(calibration.reference_side_px * MIN_BRANCH_WIDTH_REFERENCE_RATIO)
    stats = {
        "enabled": bool(ENABLE_BINARY_THIN_BRANCH_PRUNING),
        "min_width_px": round(min_width_px, 3),
        "min_width_mm": round(float(REFERENCE_SQUARE_SIZE_MM * MIN_BRANCH_WIDTH_REFERENCE_RATIO), 3),
        "reference_ratio": MIN_BRANCH_WIDTH_REFERENCE_RATIO,
        "removed_path_count": 0,
        "removed_pixel_count": 0,
    }
    if not ENABLE_BINARY_THIN_BRANCH_PRUNING or cv2.countNonZero(binary) == 0:
        return binary, stats

    distance_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    raw_skeleton = skeletonize(binary)
    paths = extract_all_paths(raw_skeleton)
    if not paths:
        return binary, stats

    degree_map = build_degree_map(raw_skeleton)
    removal_mask = np.zeros_like(binary)
    removed_paths = []

    for path_index, path in enumerate(paths, start=1):
        path_length_px = compute_smoothed_path_length(path)
        if path_length_px < MIN_THIN_BRANCH_PRUNE_LENGTH_PX:
            continue

        width_summary = compute_path_width_summary(path, distance_map, degree_map)
        mean_width_px = width_summary["mean_width_px"]
        if mean_width_px >= min_width_px:
            continue

        points_xy = np.array([(x, y) for y, x in path], dtype=np.int32).reshape(-1, 1, 2)
        thickness = max(1, int(np.ceil(min_width_px)))
        if len(points_xy) >= 2:
            cv2.polylines(removal_mask, [points_xy], False, 255, thickness, cv2.LINE_AA)
        else:
            y, x = path[0]
            cv2.circle(removal_mask, (x, y), max(1, thickness // 2), 255, -1)
        removed_paths.append({
            "path_index": path_index,
            "length_px": round(float(path_length_px), 3),
            "mean_width_px": round(float(mean_width_px), 3),
            "max_width_px": round(float(width_summary["max_width_px"]), 3),
        })

    if not removed_paths:
        return binary, stats

    pruned = binary.copy()
    before_count = int(cv2.countNonZero(pruned))
    pruned[removal_mask > 0] = 0
    after_count = int(cv2.countNonZero(pruned))

    stats["removed_path_count"] = len(removed_paths)
    stats["removed_pixel_count"] = before_count - after_count
    stats["removed_paths"] = removed_paths
    return pruned, stats


def prune_thin_crack_pixels_iterative(binary: np.ndarray, calibration: SquareCalibration):
    current = binary.copy()
    aggregate = {
        "enabled": bool(ENABLE_BINARY_THIN_BRANCH_PRUNING),
        "passes": 0,
        "min_width_px": round(float(calibration.reference_side_px * MIN_BRANCH_WIDTH_REFERENCE_RATIO), 3),
        "min_width_mm": round(float(REFERENCE_SQUARE_SIZE_MM * MIN_BRANCH_WIDTH_REFERENCE_RATIO), 3),
        "reference_ratio": MIN_BRANCH_WIDTH_REFERENCE_RATIO,
        "removed_path_count": 0,
        "removed_pixel_count": 0,
        "pass_details": [],
    }

    for pass_index in range(1, THIN_BRANCH_PRUNE_PASSES + 1):
        current, stats = prune_thin_crack_pixels(current, calibration)
        aggregate["passes"] = pass_index
        aggregate["removed_path_count"] += int(stats.get("removed_path_count", 0))
        aggregate["removed_pixel_count"] += int(stats.get("removed_pixel_count", 0))
        aggregate["pass_details"].append(stats)
        if int(stats.get("removed_path_count", 0)) == 0 and int(stats.get("removed_pixel_count", 0)) == 0:
            break

    return current, aggregate


def keep_all_crack_components(
    binary: np.ndarray,
    gray_image=None,
    blackhat_image=None,
    faint_border_rescue=False,
    dark_threshold=DARK_VALUE_THRESHOLD,
    min_keep_width_px=None,
    pixels_per_mm=None,
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    min_filter_path_length_px = scale_len(MIN_COMPONENT_FILTER_PATH_LENGTH_PX, pixels_per_mm)
    min_component_area = scale_area(MIN_COMPONENT_AREA, pixels_per_mm)
    min_rescued_length_px = scale_len(MIN_RESCUED_COMPONENT_LENGTH_PX, pixels_per_mm)

    component_ids = []
    fallback_candidates = []
    distance_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    global_skeleton = skeletonize(binary)
    global_degree_map = build_degree_map(global_skeleton)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        short_side = max(min(width, height), 1)
        aspect_ratio = max(width, height) / short_side
        fill_ratio = area / max(float(width * height), 1.0)
        touches_border = (
            x == 0
            or y == 0
            or (x + width) >= binary.shape[1]
            or (y + height) >= binary.shape[0]
        )
        component_mask = np.zeros_like(binary)
        component_mask[labels == label] = 255
        component_skeleton = skeletonize(component_mask)
        component_path = longest_centerline_path(component_skeleton)
        path_length_px = compute_smoothed_path_length(component_path)
        if path_length_px < min_filter_path_length_px:
            continue

        pca_ratio = compute_component_pca_ratio(component_mask)
        shape_pass = area >= min_component_area and (
            aspect_ratio >= MIN_COMPONENT_ASPECT_RATIO
            or pca_ratio >= MIN_COMPONENT_PCA_RATIO
            or (
                area >= min_component_area * 8
                and fill_ratio <= LARGE_COMPONENT_MAX_FILL_RATIO
                and pca_ratio >= LARGE_COMPONENT_MIN_ASPECT_RATIO * 4.0
            )
        )
        rescued_shape_pass = (
            area >= min_component_area
            and path_length_px >= min_rescued_length_px
            and fill_ratio <= MAX_RESCUED_COMPONENT_FILL_RATIO
            and pca_ratio >= MIN_RESCUED_COMPONENT_PCA_RATIO
        )
        path_q75_intensity = 0.0
        dark_pixel_ratio = 1.0
        blackhat_mean = 0.0
        can_rescue_faint_crack = False
        wide_dark_component_pass = False
        if gray_image is not None:
            path_q75_intensity = compute_path_intensity_q75(component_path, gray_image)
            component_gray = gray_image[labels == label]
            dark_pixel_ratio = float(np.mean(component_gray <= dark_threshold)) if component_gray.size else 0.0
            if blackhat_image is not None:
                component_blackhat = blackhat_image[labels == label]
                blackhat_mean = float(np.mean(component_blackhat)) if component_blackhat.size else 0.0

            can_rescue_faint_crack = (
                (ENABLE_FAINT_BORDER_RESCUE or faint_border_rescue)
                and
                rescued_shape_pass
                and touches_border
                and path_q75_intensity <= MAX_RESCUED_COMPONENT_Q75_INTENSITY
                and (
                    dark_pixel_ratio >= MIN_RESCUED_DARK_PIXEL_RATIO
                    or blackhat_mean >= MIN_RESCUED_BLACKHAT_MEAN
                )
            )

            if min_keep_width_px is not None:
                widths_px = measure_widths_for_points(component_path, distance_map, global_degree_map)
                mean_width_px = float(np.mean(widths_px)) if widths_px else 0.0
                wide_dark_component_pass = (
                    mean_width_px >= float(min_keep_width_px)
                    and path_length_px >= float(min_keep_width_px) * MIN_WIDE_DARK_COMPONENT_LENGTH_RATIO
                    and fill_ratio <= MAX_WIDE_DARK_COMPONENT_FILL_RATIO
                    and path_q75_intensity <= MAX_COMPONENT_Q75_INTENSITY
                )

            if path_q75_intensity > MAX_COMPONENT_Q75_INTENSITY and not can_rescue_faint_crack:
                fallback_candidates.append((path_length_px, label))
                continue

        fallback_candidates.append((path_length_px, label))

        if shape_pass or can_rescue_faint_crack or wide_dark_component_pass:
            component_ids.append(label)

    if not component_ids:
        fallback_candidates.sort(reverse=True)
        component_ids = [label for _, label in fallback_candidates[:1]]
    elif (ENABLE_FAINT_BORDER_RESCUE or faint_border_rescue) and fallback_candidates:
        component_lengths = {label: length for length, label in fallback_candidates}
        selected_lengths = [component_lengths[label] for label in component_ids if label in component_lengths]
        fallback_candidates.sort(reverse=True)
        best_fallback_length, best_fallback_label = fallback_candidates[0]
        if (
            best_fallback_label not in component_ids
            and selected_lengths
            and best_fallback_length >= max(selected_lengths) * 2.0
        ):
            component_ids.append(best_fallback_label)

    if not component_ids:
        return binary

    merged = np.zeros_like(binary)
    for label in component_ids:
        component_mask = np.zeros_like(binary)
        component_mask[labels == label] = 255
        merged = cv2.bitwise_or(merged, component_mask)

    bridge_size = dynamic_bridge_kernel_size(pixels_per_mm)
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_size, bridge_size))
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, bridge_kernel, iterations=2)

    return merged


def _normal_section_width_px(binary, points, idx) -> float:
    n = len(points)
    if n < 2:
        return 0.0
    i0 = max(0, idx - 1)
    i1 = min(n - 1, idx + 1)
    (y0, x0), (y1, x1) = points[i0], points[i1]
    ty, tx = float(y1 - y0), float(x1 - x0)
    norm = float(np.hypot(ty, tx))
    if norm < 1e-6:
        return 0.0
    # phap tuyen (vuong goc tiep tuyen)
    ny, nx = -tx / norm, ty / norm
    cy, cx = points[idx]
    h, w = binary.shape[:2]
    max_steps = int(max(binary.shape) )

    def march(sign):
        steps = 0
        for s in range(1, max_steps + 1):
            yy = int(round(cy + sign * ny * s))
            xx = int(round(cx + sign * nx * s))
            if yy < 0 or xx < 0 or yy >= h or xx >= w or binary[yy, xx] == 0:
                break
            steps = s
        return steps

    return float(march(1) + march(-1) + 1)


def measure_widths_for_points(points, distance_map, degree_map, binary=None):
    all_widths_px = []
    stable_widths_px = []
    ordered_points = list(points)

    for idx, (y, x) in enumerate(ordered_points):
        if distance_map[y, x] <= 0:
            continue

        width_px = float(distance_map[y, x] * 2.0)
        if binary is not None:
            normal_px = _normal_section_width_px(binary, ordered_points, idx)
            if normal_px > 0.0:
                # distance-transform de over-uoc o cho cong/giao; mat cat phap
                # tuyen de over-uoc o doan xien -> lay min cho ben & chinh xac.
                width_px = min(width_px, normal_px)
        all_widths_px.append(width_px)
        if degree_map.get((y, x), 0) == 2:
            stable_widths_px.append(width_px)

    if len(stable_widths_px) >= max(3, len(all_widths_px) // 3):
        return stable_widths_px

    return all_widths_px


def robust_max_width(widths_px):
    if not widths_px:
        return 0.0
    if len(widths_px) < 5:
        return float(np.max(widths_px))
    return float(np.percentile(widths_px, 95))


def collect_unique_path_points(paths):
    unique_points = []
    seen = set()
    for path in paths:
        for point in path:
            key = tuple(point)
            if key in seen:
                continue
            seen.add(key)
            unique_points.append(point)
    return unique_points


def smooth_path_points(path, window: int = 3):
    if len(path) < 3 or window < 2:
        return [(int(y), int(x)) for y, x in path]

    ys = np.array([float(y) for y, _ in path], dtype=np.float32)
    xs = np.array([float(x) for _, x in path], dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad = window // 2
    ys_pad = np.pad(ys, pad, mode="edge")
    xs_pad = np.pad(xs, pad, mode="edge")
    ys_s = np.convolve(ys_pad, kernel, mode="valid")[: len(ys)]
    xs_s = np.convolve(xs_pad, kernel, mode="valid")[: len(xs)]
    smoothed = [(int(round(y)), int(round(x))) for y, x in zip(ys_s, xs_s)]
    smoothed[0] = (int(path[0][0]), int(path[0][1]))
    smoothed[-1] = (int(path[-1][0]), int(path[-1][1]))
    return smoothed


def build_paths_mask(paths, shape, connect: bool = False, pixels_per_mm=None, smooth: bool = True):
    """connect=False: ban pixel roi (dung cho build_degree_map - phai chinh xac pixel).
    connect=True: ban hien thi - ve polyline lien mach + lam tron nhe + noi cac path
    co dau mut gan nhau tai nut giao, de xuong preview hoan chinh khong dut doan."""
    mask = np.zeros(shape, dtype=np.uint8)
    if not connect:
        for path in paths:
            for y, x in path:
                mask[y, x] = 255
        return mask

    drawn_paths = []
    for path in paths:
        pts = smooth_path_points(path) if smooth else [(int(y), int(x)) for y, x in path]
        drawn_paths.append(pts)
        if len(pts) >= 2:
            polyline = np.array([(x, y) for y, x in pts], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [polyline], False, 255, 1, cv2.LINE_8)
        elif pts:
            y, x = pts[0]
            mask[y, x] = 255

    if pixels_per_mm:
        join_radius = max(2, scale_px(0.3, pixels_per_mm))
        endpoints = []
        for pts in drawn_paths:
            if len(pts) >= 2:
                endpoints.append(pts[0])
                endpoints.append(pts[-1])
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                (y1, x1), (y2, x2) = endpoints[i], endpoints[j]
                dist = float(np.hypot(x2 - x1, y2 - y1))
                if 0 < dist <= join_radius:
                    cv2.line(mask, (x1, y1), (x2, y2), 255, 1, cv2.LINE_8)
    return mask


def compute_path_width_summary(path, distance_map, degree_map, binary=None):
    widths_px = measure_widths_for_points(path, distance_map, degree_map, binary=binary)
    if not widths_px:
        return {
            "mean_width_px": 0.0,
            "max_width_px": 0.0,
        }

    return {
        "mean_width_px": float(np.mean(widths_px)),
        "max_width_px": robust_max_width(widths_px),
    }


def filter_paths_by_branch_width(paths, distance_map, calibration: SquareCalibration, binary=None):
    if not paths:
        return [], {
            "min_width_px": calibration.reference_side_px * MIN_BRANCH_WIDTH_REFERENCE_RATIO,
            "min_width_mm": REFERENCE_SQUARE_SIZE_MM * MIN_BRANCH_WIDTH_REFERENCE_RATIO,
            "reference_ratio": MIN_BRANCH_WIDTH_REFERENCE_RATIO,
            "removed_branch_count": 0,
        }, []

    draft_skeleton = build_paths_mask(paths, distance_map.shape)
    degree_map = build_degree_map(draft_skeleton)
    min_width_px = calibration.reference_side_px * MIN_BRANCH_WIDTH_REFERENCE_RATIO
    kept_paths = []
    removed = []
    removed_paths = []

    for branch_id, path in enumerate(paths, start=1):
        width_summary = compute_path_width_summary(path, distance_map, degree_map, binary=binary)
        if width_summary["mean_width_px"] >= min_width_px:
            kept_paths.append(path)
            continue

        removed_paths.append(path)
        removed.append({
            "branch_id": branch_id,
            "mean_width_px": round(width_summary["mean_width_px"], 3),
            "max_width_px": round(width_summary["max_width_px"], 3),
        })

    return kept_paths, {
        "min_width_px": round(float(min_width_px), 3),
        "min_width_mm": round(float(REFERENCE_SQUARE_SIZE_MM * MIN_BRANCH_WIDTH_REFERENCE_RATIO), 3),
        "reference_ratio": MIN_BRANCH_WIDTH_REFERENCE_RATIO,
        "removed_branch_count": len(removed),
        "removed_branches": removed,
    }, removed_paths


def remove_paths_from_binary(binary: np.ndarray, paths, min_width_px: float) -> tuple[np.ndarray, int]:
    if not paths:
        return binary, 0

    removal_mask = np.zeros_like(binary)
    thickness = max(1, int(np.ceil(min_width_px)))
    for path in paths:
        points_xy = np.array([(x, y) for y, x in path], dtype=np.int32).reshape(-1, 1, 2)
        if len(points_xy) >= 2:
            cv2.polylines(removal_mask, [points_xy], False, 255, thickness, cv2.LINE_AA)
        elif len(path) == 1:
            y, x = path[0]
            cv2.circle(removal_mask, (x, y), max(1, thickness // 2), 255, -1)

    pruned = binary.copy()
    before_count = int(cv2.countNonZero(pruned))
    pruned[removal_mask > 0] = 0
    return pruned, before_count - int(cv2.countNonZero(pruned))


def extract_center_band(binary: np.ndarray) -> np.ndarray:
    image_height, image_width = binary.shape[:2]
    half_band = max(12, int(image_width * PRIMARY_CENTER_BAND_RATIO * 0.5))
    center_x = image_width // 2
    left = max(0, center_x - half_band)
    right = min(image_width, center_x + half_band + 1)

    band_mask = np.zeros_like(binary)
    band_mask[:, left:right] = 255
    return cv2.bitwise_and(binary, band_mask)


def select_primary_centerline(binary: np.ndarray, gray_image=None):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return []

    image_height, image_width = binary.shape[:2]
    image_center_x = (image_width - 1) * 0.5
    best_score = float("-inf")
    best_path = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA:
            continue

        component = np.zeros_like(binary)
        component[labels == label] = 255
        component_skeleton = skeletonize(component)
        path = longest_centerline_path(component_skeleton)
        if len(path) <= 1:
            continue

        length_px = compute_smoothed_path_length(path)
        mean_x = float(np.mean([x for _, x in path]))
        center_penalty = abs(mean_x - image_center_x) * 1.5
        darkness_bonus = 0.0
        if gray_image is not None:
            darkness_bonus = max(0.0, 180.0 - compute_path_intensity_q75(path, gray_image))
        score = length_px + darkness_bonus - center_penalty
        if score > best_score:
            best_score = score
            best_path = path

    return [best_path] if len(best_path) > 1 else []


def trace_center_crack_from_top(binary: np.ndarray):
    skeleton = skeletonize(binary)
    points = np.column_stack(np.where(skeleton > 0))
    if len(points) == 0:
        return []

    graph = build_graph(points)
    graph = {node: neighbors for node, neighbors in graph.items() if neighbors}
    if not graph:
        return []

    image_height, image_width = binary.shape[:2]
    image_center_x = (image_width - 1) * 0.5
    top_limit = max(10, int(image_height * 0.18))
    seed_candidates = [node for node in graph if node[0] <= top_limit]
    if not seed_candidates:
        seed_candidates = [min(graph, key=lambda node: node[0])]

    start = min(seed_candidates, key=lambda node: (abs(node[1] - image_center_x), node[0]))
    distances, previous = dijkstra_graph(graph, start)
    if not distances:
        return []

    def end_score(node):
        vertical_reach = node[0] - start[0]
        return (vertical_reach * 2.0) + distances[node]

    end = max(distances, key=end_score)
    path = reconstruct_graph_path(previous, start, end)
    return [path] if len(path) > 1 else []


def path_height_ratio(path, image_shape):
    if not path:
        return 0.0

    ys = [y for y, _ in path]
    image_height = max(image_shape[0] - 1, 1)
    return (max(ys) - min(ys)) / image_height


def should_use_primary_mode(primary_paths, image_shape):
    if not primary_paths:
        return False

    return path_height_ratio(primary_paths[0], image_shape) >= PRIMARY_MODE_MIN_HEIGHT_RATIO


def build_segment(index, branch_id, points, length_px, distance_map, pixels_per_mm, degree_map, binary=None):
    widths_px = measure_widths_for_points(points, distance_map, degree_map, binary=binary)
    mean_width_px = float(np.mean(widths_px)) if widths_px else 0.0
    max_width_px = robust_max_width(widths_px)

    return SegmentMeasurement(
        index=index,
        branch_id=branch_id,
        length_px=length_px,
        length_mm=length_px / pixels_per_mm,
        mean_width_px=mean_width_px,
        mean_width_mm=mean_width_px / pixels_per_mm,
        max_width_px=max_width_px,
        max_width_mm=max_width_px / pixels_per_mm,
        points=list(points),
    )


def split_path_into_segments(path, branch_id, distance_map, pixels_per_mm, segment_length_mm, start_index, degree_map, binary=None):
    if len(path) < 2:
        return [], 0.0, start_index

    target_length_px = segment_length_mm * pixels_per_mm
    segments = []
    segment_points = [path[0]]
    segment_progress_px = 0.0
    total_length_px = 0.0
    segment_index = start_index

    for idx in range(1, len(path)):
        y1, x1 = path[idx - 1]
        y2, x2 = path[idx]
        step = float(np.hypot(x2 - x1, y2 - y1))

        segment_progress_px += step
        segment_points.append((y2, x2))

        if segment_progress_px >= target_length_px:
            measured_length_px = compute_smoothed_path_length(segment_points)
            segments.append(
                build_segment(
                    index=segment_index,
                    branch_id=branch_id,
                    points=segment_points,
                    length_px=measured_length_px,
                    distance_map=distance_map,
                    pixels_per_mm=pixels_per_mm,
                    degree_map=degree_map,
                    binary=binary,
                )
            )
            total_length_px += measured_length_px
            segment_index += 1
            segment_points = [(y2, x2)]
            segment_progress_px = 0.0

    if len(segment_points) > 1:
        measured_length_px = compute_smoothed_path_length(segment_points)
        segments.append(
            build_segment(
                index=segment_index,
                branch_id=branch_id,
                points=segment_points,
                length_px=measured_length_px,
                distance_map=distance_map,
                pixels_per_mm=pixels_per_mm,
                degree_map=degree_map,
                binary=binary,
            )
        )
        total_length_px += measured_length_px
        segment_index += 1

    return segments, total_length_px, segment_index


def measure_all_paths(paths, distance_map, pixels_per_mm, segment_length_mm, degree_map, binary=None):
    all_segments = []
    total_length_px = 0.0
    next_index = 1

    for branch_id, path in enumerate(paths, start=1):
        segments, path_length_px, next_index = split_path_into_segments(
            path=path,
            branch_id=branch_id,
            distance_map=distance_map,
            pixels_per_mm=pixels_per_mm,
            segment_length_mm=segment_length_mm,
            start_index=next_index,
            degree_map=degree_map,
            binary=binary,
        )
        all_segments.extend(segments)
        total_length_px += path_length_px

    return all_segments, total_length_px


def color_for_index(index):
    palette = [
        (0, 0, 255),
        (0, 180, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 200, 0),
        (255, 0, 0),
        (255, 0, 255),
    ]
    return palette[(index - 1) % len(palette)]


def resolve_selected_segment(segments):
    if not segments:
        return None

    max_index = max(segment.index for segment in segments)
    while True:
        raw_value = input(
            f"Nhap so doan can highlight [1-{max_index}, bo trong de hien tat ca]: "
        ).strip()
        if not raw_value:
            return None

        try:
            selected_index = int(raw_value)
        except ValueError:
            print("Gia tri khong hop le. Hay nhap mot so nguyen.")
            continue

        if 1 <= selected_index <= max_index:
            return selected_index

        print(f"So doan phai nam trong khoang 1 den {max_index}.")


def draw_results(image: np.ndarray, segments, selected_index=None, show_labels=False, removed_branch_ids=None):
    overlay = image.copy()
    removed_branch_ids = removed_branch_ids or set()
    thin_branch_color = (0, 200, 255)

    for segment in segments:
        is_thin = (segment.branch_id in removed_branch_ids) or (not segment.passes_width)
        if is_thin:
            color = thin_branch_color
        elif selected_index is None:
            color = color_for_index(segment.index)
        else:
            color = (150, 150, 150)

        points_xy = np.array([(x, y) for y, x in segment.points], dtype=np.int32).reshape(-1, 1, 2)
        if len(points_xy) >= 2:
            cv2.polylines(overlay, [points_xy], False, color, 2, cv2.LINE_AA)
        else:
            for y, x in segment.points:
                overlay[y, x] = color

        if show_labels and selected_index is None and (segment.index == 1 or segment.index % LABEL_EVERY_N_SEGMENTS == 0):
            mid_y, mid_x = segment.points[len(segment.points) // 2]
            cv2.putText(
                overlay,
                str(segment.index),
                (mid_x + 4, mid_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    if selected_index is not None:
        selected_segment = next((segment for segment in segments if segment.index == selected_index), None)
        if selected_segment is not None:
            highlight_points = np.array(
                [(x, y) for y, x in selected_segment.points],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            if len(highlight_points) >= 2:
                cv2.polylines(overlay, [highlight_points], False, (0, 0, 255), 3, cv2.LINE_AA)

    return overlay


def draw_reference_overlay(image: np.ndarray, calibration: SquareCalibration):
    overlay = image.copy()
    corners_int = calibration.reference_corners.astype(np.int32)
    cv2.polylines(overlay, [corners_int], True, (0, 255, 255), 2, cv2.LINE_AA)
    center = tuple(np.mean(corners_int, axis=0).astype(int))
    cv2.putText(
        overlay,
        f"Square: {calibration.pixels_per_mm:.2f} px/mm",
        (center[0] - 72, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def point_to_web_xy(point):
    y, x = point
    return [int(x), int(y)]


def build_web_result_payload(
    image_path: Path,
    calibration: SquareCalibration,
    measurement_mode: str,
    total_length_mm: float,
    widest_segment,
    area_mm2: float,
    segments,
    input_validation=None,
    branch_width_filter=None,
    area_mm2_valid=None,
):
    widest_segment_payload = None
    if widest_segment is not None:
        widest_segment_payload = {
            "index": int(widest_segment.index),
            "branch_id": int(widest_segment.branch_id),
            "max_width_mm": round(float(widest_segment.max_width_mm), 3),
            "start_point": point_to_web_xy(widest_segment.points[0]),
            "end_point": point_to_web_xy(widest_segment.points[-1]),
        }

    return {
        "image_name": image_path.name,
        "processed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "measurement_mode": measurement_mode,
        "pixels_per_mm": round(float(calibration.pixels_per_mm), 6),
        "total_length_mm": round(float(total_length_mm), 3),
        "max_width_mm": round(float(widest_segment.max_width_mm), 3) if widest_segment is not None else 0.0,
        "area_mm2": round(float(area_mm2), 3),
        "area_mm2_valid": round(float(area_mm2_valid if area_mm2_valid is not None else area_mm2), 3),
        "segment_count": len(segments),
        "widest_segment": widest_segment_payload,
        "input_validation": input_validation or {},
        "branch_width_filter": branch_width_filter or {},
        "segments": [
            {
                "index": int(segment.index),
                "branch_id": int(segment.branch_id),
                "length_mm": round(float(segment.length_mm), 3),
                "max_width_mm": round(float(segment.max_width_mm), 3),
                "passes_width": bool(segment.passes_width),
                "start_point": point_to_web_xy(segment.points[0]),
                "end_point": point_to_web_xy(segment.points[-1]),
                "polyline": [point_to_web_xy(point) for point in segment.points],
            }
            for segment in segments
            if segment.points
        ],
    }


def export_web_result(image_path: Path, payload: dict) -> Path:
    WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = WEB_OUTPUT_DIR / f"{image_path.stem}_result.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def build_result_overlay(
    image: np.ndarray,
    segments,
    calibration: SquareCalibration,
    total_length_mm: float,
    widest_segment_max_mm: float,
    area_mm2: float,
    selected_segment=None,
    removed_branch_ids=None,
):
    selected_segment_index = selected_segment.index if selected_segment is not None else None
    return draw_results(
        image,
        segments,
        selected_index=selected_segment_index,
        show_labels=False,
        removed_branch_ids=removed_branch_ids,
    )


def export_web_artifacts(image_path: Path, binary: np.ndarray, skeleton: np.ndarray, overlay: np.ndarray):
    WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    binary_path = WEB_OUTPUT_DIR / f"{image_path.stem}_binary.jpg"
    skeleton_path = WEB_OUTPUT_DIR / f"{image_path.stem}_skeleton.jpg"
    overlay_path = WEB_OUTPUT_DIR / f"{image_path.stem}_overlay.jpg"
    cv2.imwrite(str(binary_path), binary)
    cv2.imwrite(str(skeleton_path), skeleton)
    cv2.imwrite(str(overlay_path), overlay)
    return {
        "binary_image": binary_path.name,
        "skeleton_image": skeleton_path.name,
        "overlay_image": overlay_path.name,
    }


def measure_image_core(image: np.ndarray, gray: np.ndarray, calibration: SquareCalibration, tone_profile: ToneProfile, faint_border_rescue=False):
    pixels_per_mm = calibration.pixels_per_mm
    min_width_px = float(calibration.reference_side_px * MIN_BRANCH_WIDTH_REFERENCE_RATIO)
    binary = preprocess(
        image,
        reference_corners=calibration.reference_corners,
        faint_border_rescue=faint_border_rescue,
        tone_profile=tone_profile,
        min_keep_width_px=min_width_px,
        pixels_per_mm=pixels_per_mm,
    )
    # Artifact 1 (detection binary) khong bao gio bi khoet vat ly. Prune chi
    # chay tren ban sao de lay thong ke bao cao, khong lam dut xuong/sai do rong.
    _, thin_pixel_pruning = prune_thin_crack_pixels_iterative(binary.copy(), calibration)
    measurement_binary = build_measurement_binary(binary)
    # Artifact 3 (do rong): distance transform tinh tren measurement_binary (da erode)
    # de bu tru phan phong to do MORPH_CLOSE trong preprocess.
    width_distance_map = cv2.distanceTransform(measurement_binary, cv2.DIST_L2, 5)
    paths = extract_component_centerlines(measurement_binary, pixels_per_mm, gray_image=gray)
    measurement_mode = "giu tat ca nhanh trung tam hop le trong moi vet nut"
    if faint_border_rescue:
        measurement_mode = f"{measurement_mode}; auto-rescue vet nut mo cham mep"

    if FORCE_PRIMARY_CENTER_CRACK:
        primary_binary = extract_center_band(measurement_binary)
        primary_paths = trace_center_crack_from_top(primary_binary)
        if not primary_paths:
            primary_paths = select_primary_centerline(primary_binary, gray_image=gray)
        if not primary_paths:
            primary_paths = select_primary_centerline(measurement_binary, gray_image=gray)
        if primary_paths:
            paths = primary_paths
            measurement_mode = "chi bam vet nut chinh o vung giua"
    elif len(paths) > AUTO_PRIMARY_MODE_PATH_COUNT:
        primary_binary = extract_center_band(measurement_binary)
        primary_paths = trace_center_crack_from_top(primary_binary)
        if not primary_paths:
            primary_paths = select_primary_centerline(primary_binary, gray_image=gray)
        if not primary_paths:
            primary_paths = select_primary_centerline(measurement_binary, gray_image=gray)
        if should_use_primary_mode(primary_paths, measurement_binary.shape):
            paths = primary_paths
            measurement_mode = "tu dong bam vet nut chinh o vung giua"

    if not paths:
        raise ValueError("Khong tim duoc duong trung tam hop le de do cac vet nut.")

    # Artifact 2 (xuong hien thi) GIU TAT CA path -> xuong luon hoan chinh.
    # filter_paths_by_branch_width chi DUNG DE DANH DAU nhanh khong dat do rong,
    # khong loai khoi danh sach ve va khong khoet binary.
    _kept_paths, branch_width_filter, width_removed_paths = filter_paths_by_branch_width(
        paths=paths,
        distance_map=width_distance_map,
        calibration=calibration,
        binary=measurement_binary,
    )
    removed_branch_ids = {
        int(item["branch_id"])
        for item in branch_width_filter.get("removed_branches", [])
    }
    measurement_mode = (
        f"{measurement_mode}; danh dau (khong loai bo) nhanh co do rong trung binh < "
        f"{branch_width_filter['min_width_mm']:.3f} mm "
        f"({MIN_BRANCH_WIDTH_REFERENCE_RATIO:.0%} canh o chuan)"
    )

    skeleton = build_paths_mask(
        paths, measurement_binary.shape, connect=True, pixels_per_mm=pixels_per_mm
    )
    degree_map = build_degree_map(build_paths_mask(paths, measurement_binary.shape))
    segments, total_length_px = measure_all_paths(
        paths=paths,
        distance_map=width_distance_map,
        pixels_per_mm=pixels_per_mm,
        segment_length_mm=SEGMENT_LENGTH_MM,
        degree_map=degree_map,
        binary=measurement_binary,
    )
    for segment in segments:
        if segment.branch_id in removed_branch_ids:
            segment.passes_width = False

    total_length_mm = total_length_px / pixels_per_mm
    area_px = float(cv2.countNonZero(binary))
    area_mm2 = area_px / (pixels_per_mm * pixels_per_mm)
    # Dien tich hop le = sau khi loai vung nhanh mong, CHI dung de bao cao
    # (khong anh huong xuong hien thi / do rong).
    valid_binary, _ = remove_paths_from_binary(
        binary.copy(),
        width_removed_paths,
        branch_width_filter["min_width_px"],
    )
    area_valid_px = float(cv2.countNonZero(valid_binary))
    area_mm2_valid = area_valid_px / (pixels_per_mm * pixels_per_mm)
    widest_segment = max(
        (segment for segment in segments if segment.passes_width),
        key=lambda segment: segment.mean_width_mm,
        default=max(segments, key=lambda segment: segment.mean_width_mm, default=None),
    )
    widest_segment_max_mm = widest_segment.max_width_mm if widest_segment is not None else 0.0

    overlay = build_result_overlay(
        image=image,
        segments=segments,
        calibration=calibration,
        total_length_mm=total_length_mm,
        widest_segment_max_mm=widest_segment_max_mm,
        area_mm2=area_mm2,
        selected_segment=None,
        removed_branch_ids=removed_branch_ids,
    )

    return {
        "image": image,
        "binary": binary,
        "measurement_binary": measurement_binary,
        "skeleton": skeleton,
        "path_count": len(paths),
        "segments": segments,
        "widest_segment": widest_segment,
        "widest_segment_max_mm": widest_segment_max_mm,
        "area_mm2": area_mm2,
        "area_mm2_valid": area_mm2_valid,
        "total_length_mm": total_length_mm,
        "measurement_mode": measurement_mode,
        "calibration": calibration,
        "branch_width_filter": branch_width_filter,
        "removed_branch_ids": sorted(removed_branch_ids),
        "thin_pixel_pruning": thin_pixel_pruning,
        "overlay": overlay,
    }


def should_accept_rescue(strict_analysis: dict, rescue_analysis: dict) -> bool:
    strict_length = float(strict_analysis["total_length_mm"])
    rescue_length = float(rescue_analysis["total_length_mm"])
    added_length = rescue_length - strict_length
    if added_length < AUTO_RESCUE_MIN_ADDED_LENGTH_MM:
        return False

    max_length = strict_length * (1.0 + AUTO_RESCUE_MAX_LENGTH_INCREASE_RATIO)
    if rescue_length > max_length:
        return False

    strict_segments = max(len(strict_analysis["segments"]), 1)
    rescue_segments = len(rescue_analysis["segments"])
    max_segments = strict_segments * (1.0 + AUTO_RESCUE_MAX_SEGMENT_INCREASE_RATIO)
    return rescue_segments <= max_segments


def should_accept_candidate(current_analysis: dict, candidate_analysis: dict) -> bool:
    current_length = float(current_analysis["total_length_mm"])
    candidate_length = float(candidate_analysis["total_length_mm"])
    added_length = candidate_length - current_length
    if added_length < AUTO_RESCUE_MIN_ADDED_LENGTH_MM:
        return False

    if candidate_length > current_length * (1.0 + AUTO_RESCUE_MAX_LENGTH_INCREASE_RATIO):
        return False

    current_segments = max(len(current_analysis["segments"]), 1)
    candidate_segments = len(candidate_analysis["segments"])
    if candidate_segments > current_segments * (1.0 + AUTO_RESCUE_MAX_SEGMENT_INCREASE_RATIO):
        return False

    return True


def analyze_image(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Khong doc duoc anh: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    calibration = detect_reference_square(image)
    if calibration is None:
        raise ValueError(
            "Khong tim thay o vuong den 10x10 mm hop le de hieu chuan. "
            "Hay dat o vuong den ro net, cung mat phang voi vet nut va nen xung quanh sang hon."
        )
    input_validation = validate_input_image(image, calibration)
    static_tone_profile = compute_tone_profile(gray, calibration, use_dynamic_thresholds=False)
    dynamic_tone_profile = compute_tone_profile(gray, calibration, use_dynamic_thresholds=True)

    analysis = measure_image_core(
        image=image,
        gray=gray,
        calibration=calibration,
        tone_profile=static_tone_profile,
        faint_border_rescue=False,
    )
    active_tone_profile = static_tone_profile
    threshold_mode = "static_safe"
    rescue_decision = {
        "enabled": bool(AUTO_FAINT_BORDER_RESCUE),
        "accepted": False,
        "reason": "not_run",
    }
    dynamic_threshold_decision = {
        "enabled": bool(ENABLE_DYNAMIC_DARK_THRESHOLD),
        "accepted": False,
        "reason": "not_run",
    }

    if AUTO_FAINT_BORDER_RESCUE:
        rescue_analysis = measure_image_core(
            image=image,
            gray=gray,
            calibration=calibration,
            tone_profile=static_tone_profile,
            faint_border_rescue=True,
        )
        length_increase_ratio = (
            (rescue_analysis["total_length_mm"] - analysis["total_length_mm"])
            / max(analysis["total_length_mm"], 1e-6)
        )
        segment_increase_ratio = (
            (len(rescue_analysis["segments"]) - len(analysis["segments"]))
            / max(len(analysis["segments"]), 1)
        )
        rescue_decision = {
            "enabled": True,
            "accepted": should_accept_rescue(analysis, rescue_analysis),
            "length_increase_ratio": round(float(length_increase_ratio), 6),
            "segment_increase_ratio": round(float(segment_increase_ratio), 6),
            "strict_total_length_mm": round(float(analysis["total_length_mm"]), 3),
            "rescue_total_length_mm": round(float(rescue_analysis["total_length_mm"]), 3),
        }
        rescue_decision["reason"] = "accepted" if rescue_decision["accepted"] else "rejected_by_safety_gate"
        if rescue_decision["accepted"]:
            analysis = rescue_analysis

    if ENABLE_DYNAMIC_DARK_THRESHOLD:
        dynamic_candidates = []
        for faint_border_rescue in (False, True):
            try:
                candidate = measure_image_core(
                    image=image,
                    gray=gray,
                    calibration=calibration,
                    tone_profile=dynamic_tone_profile,
                    faint_border_rescue=faint_border_rescue,
                )
            except ValueError as exc:
                dynamic_candidates.append((None, faint_border_rescue, str(exc)))
                continue
            dynamic_candidates.append((candidate, faint_border_rescue, None))

        valid_candidates = [
            (candidate, faint_border_rescue)
            for candidate, faint_border_rescue, error in dynamic_candidates
            if candidate is not None and should_accept_candidate(analysis, candidate)
        ]
        if valid_candidates:
            best_candidate, used_rescue = max(valid_candidates, key=lambda item: item[0]["total_length_mm"])
            dynamic_threshold_decision = {
                "enabled": True,
                "accepted": True,
                "reason": "accepted_by_safety_gate",
                "used_faint_border_rescue": bool(used_rescue),
                "previous_total_length_mm": round(float(analysis["total_length_mm"]), 3),
                "dynamic_total_length_mm": round(float(best_candidate["total_length_mm"]), 3),
                "length_increase_ratio": round(
                    float((best_candidate["total_length_mm"] - analysis["total_length_mm"]) / max(analysis["total_length_mm"], 1e-6)),
                    6,
                ),
                "segment_increase_ratio": round(
                    float((len(best_candidate["segments"]) - len(analysis["segments"])) / max(len(analysis["segments"]), 1)),
                    6,
                ),
            }
            analysis = best_candidate
            active_tone_profile = dynamic_tone_profile
            threshold_mode = "dynamic_accepted"
        else:
            dynamic_threshold_decision = {
                "enabled": True,
                "accepted": False,
                "reason": "rejected_by_safety_gate",
                "candidate_errors": [
                    {
                        "faint_border_rescue": bool(faint_border_rescue),
                        "error": error,
                    }
                    for candidate, faint_border_rescue, error in dynamic_candidates
                    if error is not None
                ],
            }

    artifacts = export_web_artifacts(image_path, analysis["binary"], analysis["skeleton"], analysis["overlay"])
    payload = build_web_result_payload(
        image_path=image_path,
        calibration=calibration,
        measurement_mode=analysis["measurement_mode"],
        total_length_mm=analysis["total_length_mm"],
        widest_segment=analysis["widest_segment"],
        area_mm2=analysis["area_mm2"],
        segments=analysis["segments"],
        input_validation=input_validation,
        branch_width_filter=analysis["branch_width_filter"],
        area_mm2_valid=analysis.get("area_mm2_valid"),
    )
    payload["removed_branch_ids"] = analysis.get("removed_branch_ids", [])
    payload["threshold_mode"] = threshold_mode
    payload["tone_profile"] = tone_profile_to_payload(active_tone_profile)
    payload["static_tone_profile"] = tone_profile_to_payload(static_tone_profile)
    payload["dynamic_tone_profile"] = tone_profile_to_payload(dynamic_tone_profile)
    payload["dynamic_threshold"] = dynamic_threshold_decision
    payload["thin_pixel_pruning"] = analysis["thin_pixel_pruning"]
    payload["faint_border_rescue"] = rescue_decision
    payload["artifacts"] = artifacts
    web_output_path = export_web_result(image_path, payload)

    analysis["input_validation"] = input_validation
    analysis["tone_profile"] = active_tone_profile
    analysis["dynamic_threshold"] = dynamic_threshold_decision
    analysis["faint_border_rescue"] = rescue_decision
    analysis["payload"] = payload
    analysis["web_output_path"] = web_output_path
    return analysis


def analyze_image_from_array(image: np.ndarray, image_name: str = "camera_capture.jpg") -> dict:
    """Nhan frame numpy tu camera, tra ve cung dict nhu analyze_image().

    Khac analyze_image() duy nhat o cho: nhan anh da load san (np.ndarray)
    thay vi doc tu file Path. Toan bo logic phan tich giong het.
    """
    if image is None or (hasattr(image, "size") and image.size == 0):
        raise ValueError("Anh dau vao rong hoac khong hop le.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Anh phai co 3 kenh BGR, nhan duoc shape {image.shape}."
        )

    image_path = Path(image_name)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    calibration = detect_reference_square(image)
    if calibration is None:
        raise ValueError(
            "Khong tim thay o vuong den 10x10 mm hop le de hieu chuan. "
            "Hay dat o vuong den ro net, cung mat phang voi vet nut va nen xung quanh sang hon."
        )

    input_validation = validate_input_image(image, calibration)
    static_tone_profile = compute_tone_profile(gray, calibration, use_dynamic_thresholds=False)
    dynamic_tone_profile = compute_tone_profile(gray, calibration, use_dynamic_thresholds=True)

    analysis = measure_image_core(
        image=image,
        gray=gray,
        calibration=calibration,
        tone_profile=static_tone_profile,
        faint_border_rescue=False,
    )
    active_tone_profile = static_tone_profile
    threshold_mode = "static_safe"
    rescue_decision = {
        "enabled": bool(AUTO_FAINT_BORDER_RESCUE),
        "accepted": False,
        "reason": "not_run",
    }
    dynamic_threshold_decision = {
        "enabled": bool(ENABLE_DYNAMIC_DARK_THRESHOLD),
        "accepted": False,
        "reason": "not_run",
    }

    if AUTO_FAINT_BORDER_RESCUE:
        rescue_analysis = measure_image_core(
            image=image,
            gray=gray,
            calibration=calibration,
            tone_profile=static_tone_profile,
            faint_border_rescue=True,
        )
        length_increase_ratio = (
            (rescue_analysis["total_length_mm"] - analysis["total_length_mm"])
            / max(analysis["total_length_mm"], 1e-6)
        )
        segment_increase_ratio = (
            (len(rescue_analysis["segments"]) - len(analysis["segments"]))
            / max(len(analysis["segments"]), 1)
        )
        rescue_decision = {
            "enabled": True,
            "accepted": should_accept_rescue(analysis, rescue_analysis),
            "length_increase_ratio": round(float(length_increase_ratio), 6),
            "segment_increase_ratio": round(float(segment_increase_ratio), 6),
            "strict_total_length_mm": round(float(analysis["total_length_mm"]), 3),
            "rescue_total_length_mm": round(float(rescue_analysis["total_length_mm"]), 3),
        }
        rescue_decision["reason"] = (
            "accepted" if rescue_decision["accepted"] else "rejected_by_safety_gate"
        )
        if rescue_decision["accepted"]:
            analysis = rescue_analysis

    if ENABLE_DYNAMIC_DARK_THRESHOLD:
        dynamic_candidates = []
        for faint_border_rescue in (False, True):
            try:
                candidate = measure_image_core(
                    image=image,
                    gray=gray,
                    calibration=calibration,
                    tone_profile=dynamic_tone_profile,
                    faint_border_rescue=faint_border_rescue,
                )
            except ValueError as exc:
                dynamic_candidates.append((None, faint_border_rescue, str(exc)))
                continue
            dynamic_candidates.append((candidate, faint_border_rescue, None))

        valid_candidates = [
            (candidate, faint_border_rescue)
            for candidate, faint_border_rescue, error in dynamic_candidates
            if candidate is not None and should_accept_candidate(analysis, candidate)
        ]
        if valid_candidates:
            best_candidate, used_rescue = max(
                valid_candidates, key=lambda item: item[0]["total_length_mm"]
            )
            dynamic_threshold_decision = {
                "enabled": True,
                "accepted": True,
                "reason": "accepted_by_safety_gate",
                "used_faint_border_rescue": bool(used_rescue),
                "previous_total_length_mm": round(float(analysis["total_length_mm"]), 3),
                "dynamic_total_length_mm": round(float(best_candidate["total_length_mm"]), 3),
                "length_increase_ratio": round(
                    float(
                        (best_candidate["total_length_mm"] - analysis["total_length_mm"])
                        / max(analysis["total_length_mm"], 1e-6)
                    ),
                    6,
                ),
                "segment_increase_ratio": round(
                    float(
                        (len(best_candidate["segments"]) - len(analysis["segments"]))
                        / max(len(analysis["segments"]), 1)
                    ),
                    6,
                ),
            }
            analysis = best_candidate
            active_tone_profile = dynamic_tone_profile
            threshold_mode = "dynamic_accepted"
        else:
            dynamic_threshold_decision = {
                "enabled": True,
                "accepted": False,
                "reason": "rejected_by_safety_gate",
                "candidate_errors": [
                    {"faint_border_rescue": bool(faint_border_rescue), "error": error}
                    for candidate, faint_border_rescue, error in dynamic_candidates
                    if error is not None
                ],
            }

    artifacts = export_web_artifacts(
        image_path, analysis["binary"], analysis["skeleton"], analysis["overlay"]
    )
    payload = build_web_result_payload(
        image_path=image_path,
        calibration=calibration,
        measurement_mode=analysis["measurement_mode"],
        total_length_mm=analysis["total_length_mm"],
        widest_segment=analysis["widest_segment"],
        area_mm2=analysis["area_mm2"],
        segments=analysis["segments"],
        input_validation=input_validation,
        branch_width_filter=analysis["branch_width_filter"],
        area_mm2_valid=analysis.get("area_mm2_valid"),
    )
    payload["removed_branch_ids"] = analysis.get("removed_branch_ids", [])
    payload["threshold_mode"] = threshold_mode
    payload["tone_profile"] = tone_profile_to_payload(active_tone_profile)
    payload["static_tone_profile"] = tone_profile_to_payload(static_tone_profile)
    payload["dynamic_tone_profile"] = tone_profile_to_payload(dynamic_tone_profile)
    payload["dynamic_threshold"] = dynamic_threshold_decision
    payload["thin_pixel_pruning"] = analysis["thin_pixel_pruning"]
    payload["faint_border_rescue"] = rescue_decision
    payload["artifacts"] = artifacts
    web_output_path = export_web_result(image_path, payload)

    analysis["input_validation"] = input_validation
    analysis["tone_profile"] = active_tone_profile
    analysis["dynamic_threshold"] = dynamic_threshold_decision
    analysis["faint_border_rescue"] = rescue_decision
    analysis["payload"] = payload
    analysis["web_output_path"] = web_output_path
    return analysis
