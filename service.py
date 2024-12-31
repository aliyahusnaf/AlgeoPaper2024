import numpy as np
from PIL import Image
import csv

def cosine_similarity(vec1, vec2, epsilon=1e-8):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1) + epsilon
    norm_vec2 = np.linalg.norm(vec2) + epsilon
    return dot_product / (norm_vec1 * norm_vec2)

def grayscaling(image):
    image_array = np.array(image)
    if len(image_array.shape) == 3:  # RGB to grayscale
        R, G, B = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
        grayscale = 0.2989 * R + 0.5870 * G + 0.1140 * B
    else:  # Grayscale already
        grayscale = image_array
    return grayscale

def extract_minutiae(image):
    skeleton = (image > 100).astype(np.uint8)  # Thresholding 
    minutiae_points = []

    # 3x3 kernel for Crossing Number
    rows, cols = skeleton.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j] == 1:  # Check ridge pixel only
                neighbors = [
                    skeleton[i-1, j], skeleton[i-1, j+1], skeleton[i, j+1], skeleton[i+1, j+1],
                    skeleton[i+1, j], skeleton[i+1, j-1], skeleton[i, j-1], skeleton[i-1, j-1]
                ]
                cn = sum((neighbors[k] - neighbors[k-1]) == 1 for k in range(8)) + (neighbors[0] - neighbors[-1] == 1)
                if cn == 1 or cn == 3:  # Ridge ending or bifurcation
                    minutiae_points.append((i, j))
    return minutiae_points

def minutiae_points_to_vector(minutiae_points, vector_size=120):
    vector = np.zeros(vector_size)
    for x, y in minutiae_points:
        index = (x * 16 + y) % vector_size
        vector[int(index)] = 1
    return vector

def process_query_image(query_image_path, target_size=(16, 16)):
    image = Image.open(query_image_path).convert("L").resize(target_size)
    minutiae_points = extract_minutiae(np.array(image))
    return minutiae_points_to_vector(minutiae_points)

def load_library(csv_path):
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    mean_vector = np.array([float(x) for x in rows[0][1:]])
    eigenvectors = np.array([[float(x) for x in row[1:]] for row in rows[2:122]])
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0)
    projections = np.array([[float(x) for x in row[1:]] for row in rows[133:]])
    image_names = [row[0] for row in rows[133:]]

    return mean_vector, eigenvectors, projections, image_names

def query_minutiae(query_vector, mean_vector, eigenvectors, 
                   projections, image_names):
    standardized_query = query_vector - mean_vector
    projected_query = np.dot(standardized_query, eigenvectors)

    similarities = []
    for name, projection in zip(image_names, projections):
        similarity = cosine_similarity(projection, projected_query)
        similarities.append((name, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

if __name__ == "__main__":
    library_path = "library.csv"
    query_image_path = "/Users/aliyahusnafayyaza/Documents/MakalahAlgeo/train_data/00008_05.bmp"

    mean_vector, eigenvectors, projections, image_names = load_library(library_path)
    query_vector = process_query_image(query_image_path)

    results = query_minutiae(query_vector, mean_vector, eigenvectors, projections, image_names)
    print("Top matches:")
    for name, similarity in results[:5]:
        print(f"{name}: {similarity:.4f}")
