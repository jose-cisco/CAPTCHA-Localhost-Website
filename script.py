from PIL import Image, ImageDraw
import os
import random
import math
import numpy as np
import cv2
from skimage.util import random_noise
from sklearn.cluster import KMeans

def calculate_centroid(points):
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    centroid_x = sum(x_coords) / len(points)
    centroid_y = sum(y_coords) / len(points)
    return (centroid_x, centroid_y)

def euclid_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def generate_point(existing_centroids, width, height, overlap_probability=0):
    if not existing_centroids:
        return (random.randint(width // 8, 3 * width // 8), random.randint(height // 8, 3 * height // 8))

    avg_centroid = calculate_centroid(existing_centroids)
    max_offset = min(width, height) // 4
    max_attempts = 8

    for _ in range(max_attempts):
        new_point = (
            random.randint(int(avg_centroid[0] + max_offset), int(avg_centroid[0] + max_offset)),
            random.randint(int(avg_centroid[1] + max_offset), int(avg_centroid[1] + max_offset))
        )

        if overlap_probability > 0:
            too_close = any(euclid_distance(new_point, centroid) < max_offset * overlap_probability for centroid in existing_centroids)
            if not too_close:
                return new_point
        else:
            return new_point

    return (0, 0)

def draw_random_rotated_rectangle(draw, center, w, h):
    center_x, center_y = center
    width = random.randint(50, 130)
    height = random.randint(50, 130)
    angle = random.uniform(0, 360)
    half_width = width // 2
    half_height = height // 2
    points = [
        (center_x - half_width, center_y - half_height),
        (center_x + half_width, center_y - half_height),
        (center_x + half_width, center_y + half_height),
        (center_x - half_width, center_y + half_height),
    ]
    rotated_points = [(x, y) for x, y in points]
    for i in range(4):
        rotated_points[i] = (
            center_x + math.cos(math.radians(angle)) * (points[i][0] - center_x) - math.sin(math.radians(angle)) * (points[i][1] - center_y),
            center_y + math.sin(math.radians(angle)) * (points[i][0] - center_x) + math.cos(math.radians(angle)) * (points[i][1] - center_y)
        )
    shape_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    draw.polygon(rotated_points, fill=shape_color)

def draw_random_triangle(draw, center, w, h):
    center_x, center_y = center
    base = random.randint(50, 130)
    height = random.randint(50, 130)
    angle = random.uniform(0, 360)
    angle_radians = math.radians(angle)
    top_vertex = (center_x, center_y - height / 2)
    left_vertex = (center_x - base / 2, center_y + height / 2)
    right_vertex = (center_x + base / 2, center_y + height / 2)

    rotated_top = (
        center_x + math.cos(angle_radians) * (top_vertex[0] - center_x) - math.sin(angle_radians) * (top_vertex[1] - center_y),
        center_y + math.sin(angle_radians) * (top_vertex[0] - center_x) + math.cos(angle_radians) * (top_vertex[1] - center_y)
    )
    rotated_left = (
        center_x + math.cos(angle_radians) * (left_vertex[0] - center_x) - math.sin(angle_radians) * (left_vertex[1] - center_y),
        center_y + math.sin(angle_radians) * (left_vertex[0] - center_x) + math.cos(angle_radians) * (left_vertex[1] - center_y)
    )
    rotated_right = (
        center_x + math.cos(angle_radians) * (right_vertex[0] - center_x) - math.sin(angle_radians) * (right_vertex[1] - center_y),
        center_y + math.sin(angle_radians) * (right_vertex[0] - center_x) + math.cos(angle_radians) * (right_vertex[1] - center_y)
    )

    shape_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    draw.polygon([rotated_top, rotated_left, rotated_right], fill=shape_color)

# Function to apply quantization using k-means clustering
def apply_quantization(image, k=0, gaussian_blur_size=0, gaussian_blur_sigma=0,
                       gaussian_noise_std=0, poisson_noise=True, salt_and_pepper_amount=0):
    smoothed_image = cv2.GaussianBlur(image, (gaussian_blur_size, gaussian_blur_size), gaussian_blur_sigma)
    noisy_image = random_noise(smoothed_image, mode='gaussian', var=gaussian_noise_std**2)

    if poisson_noise:
        noisy_image = random_noise(noisy_image, mode='poisson', seed=None, clip=True)

    noisy_image = random_noise(noisy_image, mode='s&p', amount=salt_and_pepper_amount)

    height, width, channels = noisy_image.shape
    flattened_image = noisy_image.reshape((height * width, channels))

    kmeans = KMeans(n_clusters=k, random_state=0)
    quantized_colors = kmeans.fit_predict(flattened_image)

    quantized_image = kmeans.cluster_centers_[quantized_colors]
    quantized_image = quantized_image.reshape((height, width, channels))

    return quantized_image

def save_processing_image(quantized_image, output_folder, img_name):
    output_path = os.path.join(output_folder, f"processing_{img_name}.png")
    scaled_image = (quantized_image * 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(scaled_image, cv2.COLOR_RGB2BGR))
    print(f"Quantized image saved to {output_path}")

if __name__ == "__main__":
    output_folder = r'img'
    os.makedirs(output_folder, exist_ok=True)

# Parameters for quantization
k_value = 16
gaussian_blur_size_value = 19
gaussian_blur_sigma_value = 19
gaussian_noise_std_value = 0.19
poisson_noise_scale_value = 0.19
salt_and_pepper_amount_value = 0.19

def gen_save_img(img_name, output_folder):
    width, height = 600, 400
    background_color = (255, 255, 255)
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)
    num_rect = 0
    num_tri = 0
    num_shapes = random.randint(3, 5)
    center_point = []

    for _ in range(num_shapes):
        new_point = generate_point(center_point, width, height, overlap_probability=0.01)
        center_point.append(new_point)

        if random.choice([True, False]):
            draw_random_rotated_rectangle(draw, new_point, width, height)
            num_rect += 1
        else:
            draw_random_triangle(draw, new_point, width, height)
            num_tri += 1

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
# Save only the processing image without keeping the original
    image.save(os.path.join(output_folder, f"processing_{img_name}.png"))
    img_path = os.path.join(output_folder, f"processing_{img_name}.png")
    img = cv2.imread(img_path)

    processing_image = apply_quantization(img, k_value, gaussian_blur_size_value,
                                  gaussian_blur_sigma_value, gaussian_noise_std_value,
                                  poisson_noise_scale_value, salt_and_pepper_amount_value)

    save_processing_image(processing_image, output_folder, img_name)
    return num_tri, num_rect
    