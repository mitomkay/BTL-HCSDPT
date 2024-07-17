from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances


app = Flask(__name__)

# Trích xuất đặc trưng về màu sắc của ảnh
def extract_color_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv_image], [0, 1, 2], None, [12, 12, 3], [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Tạo các đặc trưng ảnh và nhãn
features = []
labels = []

# Load images and extract features
data_new_path = "data_new"
for label in os.listdir(data_new_path):
    label_path = os.path.join(data_new_path, label)
    if os.path.isdir(label_path):
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (350, 700))
            feature = extract_color_features(image)
            features.append(feature)
            labels.append(label)

# Chuyển đổi về dạng mảng 
features = np.array(features)
labels = np.array(labels)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Sử dụng KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)

# Tìm kiếm ảnh gần nhất
def find_nearest_images(image_features, features, labels, k=3):
    distances = euclidean_distances(image_features, features)
    nearest_indices = np.argsort(distances)[0][:k]
    nearest_images = [(labels[i], f"data_new/{labels[i]}/{os.listdir(f'data_new/{labels[i]}')[0]}") for i in nearest_indices]
    return nearest_images

# Tính toán độ chính xác khi đưa ra ảnh gần nhất với ảnh đầu vào
def calculate_accuracy(nearest_images, true_label):
    nearest_labels = [label for label, _ in nearest_images]
    accuracy = nearest_labels.count(true_label) / len(nearest_labels)
    return accuracy

# Tìm kiếm ảnh
@app.route('/', methods=['GET'])
def predict():
    return render_template('predict.html', prediction=None)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    file = request.files['file']
    if file:
        # Lưu file ảnh mới vào thư mục test2
        file_path = "test2/input.jpg"
        file.save(file_path)

        # Trích xuất đặc trưng về màu sắc của ảnh đầu vào
        image = cv2.imread(file_path)
        image = cv2.resize(image, (350, 700))
        image_features = extract_color_features(image).reshape(1, -1)

        # Dự đoán nhãn của ảnh đầu vào
        predicted_label = knn_model.predict(image_features)[0]

        # Tìm ảnh gần nhất
        nearest_images = find_nearest_images(image_features, x_train, y_train)


        # Tính toán độ chính xác khi đưa ra ảnh gần nhất với ảnh đầu vào
        true_label = predicted_label  # Sử dụng nhãn dự đoán làm nhãn thực tế
        accuracy = calculate_accuracy(nearest_images, true_label)

        return render_template('result.html', prediction=predicted_label, nearest_images=nearest_images, accuracy=accuracy)
    else:
        return jsonify({'error': 'Không có ảnh nào được tìm thấy!'})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
