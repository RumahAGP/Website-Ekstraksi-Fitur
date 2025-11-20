let selectedFile = null;

const imageInput = document.getElementById('imageInput');
const processBtn = document.getElementById('processBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

// Event Listeners
imageInput.addEventListener('change', handleFileSelect);
processBtn.addEventListener('click', processImage);

function handleFileSelect(event) {
    selectedFile = event.target.files[0];
    if (selectedFile) {
        processBtn.disabled = false;
        
        const uploadText = document.querySelector('.upload-text');
        uploadText.textContent = `✅ ${selectedFile.name}`;
        uploadText.style.color = '#48bb78';
        
        const uploadHint = document.querySelector('.upload-hint');
        const fileSizeKB = (selectedFile.size / 1024).toFixed(2);
        uploadHint.textContent = `Ukuran: ${fileSizeKB} KB`;
    }
}

async function processImage() {
    if (!selectedFile) {
        alert('⚠️ Pilih gambar terlebih dahulu!');
        return;
    }
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    processBtn.disabled = true;
    
    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Gagal memproses gambar');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        alert('❌ Error: ' + error.message);
        console.error('Error:', error);
    } finally {
        loading.classList.add('hidden');
        processBtn.disabled = false;
    }
}

function displayResults(data) {
    // Image Info
    const info = data.image_info;
    document.getElementById('imageInfo').innerHTML = `
        <div><strong>Filename:</strong> ${info.filename}</div>
        <div><strong>Dimensi:</strong> ${info.size}</div>
        <div><strong>Channels:</strong> ${info.channels}</div>
        <div><strong>File Size:</strong> ${info.file_size}</div>
    `;
    
    // Original Image
    document.getElementById('originalImage').src = data.image_url;
    
    // 1. Grayscale
    document.getElementById('desc-grayscale').textContent = data.grayscale.description;
    document.getElementById('img-grayscale').src = 'data:image/png;base64,' + data.grayscale.image;
    document.getElementById('hist-grayscale').src = 'data:image/png;base64,' + data.grayscale.histogram;
    
    // 2. Histogram Equalization
    document.getElementById('desc-equalization').textContent = data.histogram_equalization.description;
    document.getElementById('img-eq-before').src = 'data:image/png;base64,' + data.histogram_equalization.original;
    document.getElementById('img-eq-after').src = 'data:image/png;base64,' + data.histogram_equalization.equalized;
    document.getElementById('hist-eq-before').src = 'data:image/png;base64,' + data.histogram_equalization.histogram_before;
    document.getElementById('hist-eq-after').src = 'data:image/png;base64,' + data.histogram_equalization.histogram_after;
    
    // 3. Canny
    document.getElementById('desc-canny').textContent = data.canny.description;
    document.getElementById('img-canny-orig').src = 'data:image/png;base64,' + data.canny.original;
    document.getElementById('img-canny-1').src = 'data:image/png;base64,' + data.canny.canny_50_150;
    document.getElementById('img-canny-2').src = 'data:image/png;base64,' + data.canny.canny_100_200;
    document.getElementById('img-canny-3').src = 'data:image/png;base64,' + data.canny.canny_150_250;
    
    // 4. Sobel
    document.getElementById('desc-sobel').textContent = data.sobel.description;
    document.getElementById('img-sobel-x').src = 'data:image/png;base64,' + data.sobel.sobel_x;
    document.getElementById('img-sobel-y').src = 'data:image/png;base64,' + data.sobel.sobel_y;
    document.getElementById('img-sobel-mag').src = 'data:image/png;base64,' + data.sobel.sobel_magnitude;
    
    // 5. Thresholding
    document.getElementById('desc-threshold').textContent = data.thresholding.description;
    document.getElementById('img-thresh-bin').src = 'data:image/png;base64,' + data.thresholding.binary;
    document.getElementById('img-thresh-inv').src = 'data:image/png;base64,' + data.thresholding.binary_inv;
    document.getElementById('img-thresh-otsu').src = 'data:image/png;base64,' + data.thresholding.otsu;
    document.getElementById('img-thresh-adapt-mean').src = 'data:image/png;base64,' + data.thresholding.adaptive_mean;
    document.getElementById('img-thresh-adapt-gauss').src = 'data:image/png;base64,' + data.thresholding.adaptive_gaussian;
    
    // 6. Segmentation
    document.getElementById('desc-segmentation').textContent = data.segmentation.description;
    document.getElementById('img-seg-binary').src = 'data:image/png;base64,' + data.segmentation.binary;
    document.getElementById('img-seg-contours').src = 'data:image/png;base64,' + data.segmentation.contours;
    document.getElementById('img-seg-watershed').src = 'data:image/png;base64,' + data.segmentation.watershed;
    document.getElementById('img-seg-kmeans').src = 'data:image/png;base64,' + data.segmentation.kmeans;
    document.getElementById('contour-count').textContent = `Jumlah Objek: ${data.segmentation.num_contours}`;
    
    // Show results
    results.classList.remove('hidden');
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

console.log('✅ Image Processing App - Ready!');