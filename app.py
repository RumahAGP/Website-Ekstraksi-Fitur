from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import os
import base64
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

os.makedirs('static/uploads', exist_ok=True)

# ===================================
# HELPER FUNCTIONS
# ===================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(img):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def plot_histogram(img, title="Histogram"):
    """Create histogram plot and return base64"""
    plt.figure(figsize=(8, 4))
    
    if len(img.shape) == 2:  # Grayscale
        plt.hist(img.ravel(), bins=256, range=[0, 256], color='gray')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title(title)
    else:  # Color
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend(['Blue', 'Green', 'Red'])
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return plot_base64

# ===================================
# FEATURE EXTRACTION FUNCTIONS
# ===================================

def extract_grayscale(image_path):
    """Konversi ke Grayscale"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return {
        'image': image_to_base64(gray),
        'histogram': plot_histogram(gray, "Histogram Grayscale"),
        'description': 'Konversi gambar RGB ke Grayscale menggunakan formula: Gray = 0.299*R + 0.587*G + 0.114*B'
    }

def extract_histogram_equalization(image_path):
    """Histogram Equalization"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Histogram equalization
    equalized = cv2.equalizeHist(gray)
    
    # Create comparison
    comparison = np.hstack((gray, equalized))
    
    return {
        'original': image_to_base64(gray),
        'equalized': image_to_base64(equalized),
        'comparison': image_to_base64(comparison),
        'histogram_before': plot_histogram(gray, "Histogram Sebelum Equalization"),
        'histogram_after': plot_histogram(equalized, "Histogram Setelah Equalization"),
        'description': 'Histogram Equalization meningkatkan kontras gambar dengan mendistribusikan ulang nilai pixel secara merata'
    }

def extract_canny(image_path):
    """Canny Edge Detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Multiple thresholds
    canny_50_150 = cv2.Canny(gray, 50, 150)
    canny_100_200 = cv2.Canny(gray, 100, 200)
    canny_150_250 = cv2.Canny(gray, 150, 250)
    
    return {
        'original': image_to_base64(gray),
        'canny_50_150': image_to_base64(canny_50_150),
        'canny_100_200': image_to_base64(canny_100_200),
        'canny_150_250': image_to_base64(canny_150_250),
        'description': 'Canny Edge Detection mendeteksi tepi menggunakan gradient dan non-maximum suppression. Threshold rendah = lebih banyak detail, threshold tinggi = tepi utama saja.'
    }

def extract_sobel(image_path):
    """Sobel Edge Detection"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sobel X dan Y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = np.uint8(sobel_magnitude)
    
    # Absolute values
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)
    
    # Combined
    sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)
    
    return {
        'original': image_to_base64(gray),
        'sobel_x': image_to_base64(sobel_x_abs),
        'sobel_y': image_to_base64(sobel_y_abs),
        'sobel_magnitude': image_to_base64(sobel_magnitude),
        'sobel_combined': image_to_base64(sobel_combined),
        'description': 'Sobel mendeteksi tepi dengan menghitung gradient horizontal (X) dan vertikal (Y). Magnitude = akar(X² + Y²)'
    }

def extract_thresholding(image_path):
    """Thresholding / Binarisasi"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binary thresholding
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Otsu's thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Adaptive thresholding
    adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
    adaptive_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
    
    return {
        'original': image_to_base64(gray),
        'binary': image_to_base64(binary),
        'binary_inv': image_to_base64(binary_inv),
        'otsu': image_to_base64(otsu),
        'adaptive_mean': image_to_base64(adaptive_mean),
        'adaptive_gaussian': image_to_base64(adaptive_gaussian),
        'description': 'Thresholding mengubah gambar menjadi hitam-putih berdasarkan nilai ambang. Otsu mencari threshold optimal otomatis. Adaptive threshold menggunakan nilai lokal.'
    }

def extract_segmentation(image_path):
    """Segmentasi Citra"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Thresholding
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 3. Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    img_contours = img_rgb.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    
    # 4. Watershed segmentation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    img_watershed = img.copy()
    markers_result = cv2.watershed(img_watershed, markers)
    img_watershed[markers_result == -1] = [0, 255, 0]
    img_watershed_rgb = cv2.cvtColor(img_watershed, cv2.COLOR_BGR2RGB)
    
    # 5. K-Means segmentation
    pixels = img_rgb.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented_img = segmented.reshape(img_rgb.shape)
    
    return {
        'original': image_to_base64(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)),
        'binary': image_to_base64(binary),
        'contours': image_to_base64(cv2.cvtColor(img_contours, cv2.COLOR_RGB2BGR)),
        'watershed': image_to_base64(cv2.cvtColor(img_watershed_rgb, cv2.COLOR_RGB2BGR)),
        'kmeans': image_to_base64(cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR)),
        'num_contours': len(contours),
        'description': 'Segmentasi membagi gambar menjadi beberapa region. Contour untuk deteksi objek, Watershed untuk pemisahan objek yang overlap, K-Means untuk segmentasi berbasis warna.'
    }

# ===================================
# ROUTES
# ===================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get image info
            img = cv2.imread(filepath)
            height, width = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1
            file_size = os.path.getsize(filepath) / 1024  # KB
            
            # Process all features
            results = {
                'success': True,
                'image_url': f'/static/uploads/{filename}',
                'image_info': {
                    'filename': file.filename,
                    'size': f"{width} x {height}",
                    'channels': channels,
                    'file_size': f"{file_size:.2f} KB"
                },
                'grayscale': extract_grayscale(filepath),
                'histogram_equalization': extract_histogram_equalization(filepath),
                'canny': extract_canny(filepath),
                'sobel': extract_sobel(filepath),
                'thresholding': extract_thresholding(filepath),
                'segmentation': extract_segmentation(filepath)
            }
            
            return jsonify(results)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🖼️  Image Processing - Feature Extraction")
    print("="*60)
    print("📚 Materi:")
    print("  1. Histogram Grayscale")
    print("  2. Histogram Equalization")
    print("  3. Canny Edge Detection")
    print("  4. Sobel Edge Detection")
    print("  5. Thresholding")
    print("  6. Segmentasi Citra")
    print("="*60)
    print("🌐 Running on: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)