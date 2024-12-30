import numpy as np
from PIL import Image
import os
import csv
import primitifmatriks

# Konfigurasi
TARGET_SIZE = (16, 16)  # Ukuran gambar setelah resize
DATASET_PATH = "/Users/aliyahusnafayyaza/Documents/MakalahAlgeo/train_data"  # Ganti dengan path folder dataset
OUTPUT_FILE = "library.csv"  # File output hasil ekstraksi

# Fungsi grayscaling
def grayscaling(image):
    image_array = np.array(image)
    if len(image_array.shape) == 3:  # RGB ke grayscale
        R, G, B = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
        grayscale = 0.2989 * R + 0.5870 * G + 0.1140 * B
    else:  # Sudah grayscale
        grayscale = image_array
    return grayscale

def extract_minutiae(image):
    skeleton = (image > 100).astype(np.uint8)  
    minutiae_points = []

    rows, cols = skeleton.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j] == 1:  
                neighbors = [
                    skeleton[i-1, j], skeleton[i-1, j+1], skeleton[i, j+1], skeleton[i+1, j+1],
                    skeleton[i+1, j], skeleton[i+1, j-1], skeleton[i, j-1], skeleton[i-1, j-1]
                ]
                cn = sum((neighbors[k] - neighbors[k-1]) == 1 for k in range(8)) + (neighbors[0] - neighbors[-1] == 1)
                if cn == 1 or cn == 3: 
                    minutiae_points.append((i, j))
    return minutiae_points

# Load dataset
def load_dataset(folder_path, target_size=TARGET_SIZE):
    data = []
    image_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert("L")
            image = image.resize(target_size)

            minutiae_points = extract_minutiae(np.array(image))
            data.append(minutiae_points_to_vector(minutiae_points))
            image_names.append(filename)

    return np.array(data), image_names

def minutiae_points_to_vector(minutiae_points, vector_size=120):
    vector = np.zeros(vector_size)
    for x, y in minutiae_points:
        index = (x * 16 + y) % vector_size
        vector[int(index)] = 1
    return vector


def svd_matching(data):
    mean_vector = np.mean(data, axis=0)
    standardized_data = data - mean_vector

    covariance = np.dot(standardized_data.T, standardized_data) / len(data)

    eigenvectors = primitifmatriks.svd(covariance) 
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0)
    
    projections = np.dot(standardized_data, eigenvectors)
    return mean_vector, eigenvectors, projections

def save_to_csv(mean_vector, eigenvectors, projections, image_names, output_file=OUTPUT_FILE):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["Mean_Vector"] + mean_vector.tolist())

        writer.writerow(["Eigenvectors"])
        for eigenvector in eigenvectors:
            writer.writerow([""] + eigenvector.tolist())

        writer.writerow(["Projections"])
        for name, projection in zip(image_names, projections):
            writer.writerow([name] + projection.tolist())

# Main workflow
if __name__ == "__main__":
    print("Loading dataset...")
    data, image_names = load_dataset(DATASET_PATH, TARGET_SIZE)
    print(f"Loaded {len(data)} images.")

    print("Performing SVD Matching...")
    mean_vector, eigenvectors, projections = svd_matching(data)

    print("Saving results...")
    save_to_csv(mean_vector, eigenvectors, projections, image_names, OUTPUT_FILE)
    print(f"Results saved to {OUTPUT_FILE}.")
