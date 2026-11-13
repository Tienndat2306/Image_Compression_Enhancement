# app.py
from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import os, io, time, uuid
import cv2

app = Flask(__name__)
app.secret_key = 'pca_image_compression_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'jfif'}

def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_sharpness(img_array):
    """Tính độ sắc nét dựa trên phương sai Laplacian (Laplacian variance)"""
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def enhance_image(img_array, method='combined', strength=2):
    """
    Làm rõ / tăng nét ảnh bằng OpenCV & Pillow
    - img_array: numpy array RGB uint8
    - method: 'combined', 'unsharp', 'clahe', 'laplacian'
    - strength: 1, 2, 3, 4, 5
    """
    strength_map = {1: 0.8, 2: 1.4, 3: 2.0, 4: 2.8, 5: 3.8}
    weight = strength_map.get(strength, 1.4)
    img = img_array.copy()
    
    if method == 'unsharp':
        sigma = 1.0 + (strength * 0.15)
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
        sharpened = cv2.addWeighted(img, 1.0 + weight, blurred, -weight, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
        
    elif method == 'clahe':
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clip_limit = 1.2 + (strength * 0.8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        
    elif method == 'combined':
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clip_limit = 1.0 + (strength * 0.6)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        contrast_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        
        blurred = cv2.GaussianBlur(contrast_img, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(contrast_img, 1.0 + (weight * 0.75), blurred, -(weight * 0.75), 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
        
    elif method == 'laplacian':
        alpha = weight * 0.8
        kernel = np.array([
            [0, -alpha, 0],
            [-alpha, 1 + 4 * alpha, -alpha],
            [0, -alpha, 0]
        ], dtype=np.float32)
        filtered = cv2.filter2D(img, -1, kernel)
        return np.clip(filtered, 0, 255).astype(np.uint8)
        
    return img

METHOD_NAMES = {
    'combined': 'Kết hợp (Tương phản CLAHE + Sắc nét Unsharp Mask)',
    'unsharp': 'Làm nét chi tiết (Unsharp Masking)',
    'clahe': 'Tăng tương phản thích ứng (CLAHE)',
    'laplacian': 'Tăng cường viền cạnh (Bộ lọc Laplacian)'
}

STRENGTH_LABELS = {
    1: 'Nhẹ',
    2: 'Vừa phải (Khuyên dùng)',
    3: 'Sắc nét',
    4: 'Mạnh',
    5: 'Tối đa'
}

def convert_to_rgb(img):
    """Xử lý an toàn ảnh có kênh trong suốt (Alpha/RGBA) sang RGB nền trắng"""
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        bg = Image.new('RGB', rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return img.convert('RGB')

@app.route('/', methods=['GET', 'POST'])
def upload():
    result = None
    if request.method == 'POST':
        mode = request.form.get('mode', 'pca')
        file = request.files.get('file')
        
        if not file or file.filename == '':
            flash('Vui lòng chọn một file ảnh trước khi thực hiện!', 'warning')
            return render_template('index.html', result=None)
            
        if not allowed_file(file.filename):
            flash('Định dạng file không được hỗ trợ! Vui lòng chọn ảnh JPG, PNG, WEBP, BMP.', 'danger')
            return render_template('index.html', result=None)
            
        try:
            # Tạo tên file độc nhất tránh xung đột đa người dùng
            uid = uuid.uuid4().hex[:8]
            safe_name = secure_filename(file.filename)
            filename = f"{uid}_{safe_name}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Đo kích thước file gốc
            original_size_bytes = os.path.getsize(filepath)
            original_size_kb = original_size_bytes / 1024
            original_size_display = f"{original_size_kb/1024:.2f} MB" if original_size_kb > 1024 else f"{original_size_kb:.2f} KB"
            
            # Đọc ảnh an toàn
            img = Image.open(filepath)
            img = convert_to_rgb(img)
            arr = np.array(img)
            
            H, W = arr.shape[0], arr.shape[1]
            max_k = min(H, W)
            
            if mode == 'pca':
                raw_k = int(request.form.get('k', 50))
                # Ngăn ngừa lỗi crash khi k vượt quá kích thước ảnh
                adjusted_k = False
                if raw_k > max_k:
                    k = max_k
                    adjusted_k = True
                elif raw_k < 1:
                    k = 1
                else:
                    k = raw_k
                    
                start = time.time()
                # Xử lý PCA
                if arr.ndim == 2:
                    data = arr.astype(float)
                    pca = PCA(n_components=k)
                    comp = pca.fit_transform(data)
                    recon = pca.inverse_transform(comp)
                    recon = np.clip(recon, 0, 255).astype(np.uint8)
                else:
                    recon = np.zeros_like(arr)
                    for ch in range(3):
                        channel = arr[:, :, ch].astype(float)
                        pca = PCA(n_components=k)
                        comp = pca.fit_transform(channel)
                        ch_inv = pca.inverse_transform(comp)
                        recon[:, :, ch] = np.clip(ch_inv, 0, 255)
                    recon = recon.astype(np.uint8)
                elapsed = time.time() - start
                
                # Tính MSE, PSNR
                orig = arr.astype(float)
                rec = recon.astype(float)
                mse = np.mean((orig - rec) ** 2)
                psnr = 10 * np.log10((255.0 ** 2) / mse) if mse > 0 else float('inf')
                
                # Lưu ảnh đã nén
                outname = 'recon_' + filename
                outpath = os.path.join(app.config['UPLOAD_FOLDER'], outname)
                # Lưu chất lượng cao tối ưu để tránh sai lệch nén file
                Image.fromarray(recon).save(outpath, quality=92)
                
                # Kích thước file thực tế trên ổ đĩa
                compressed_size_bytes = os.path.getsize(outpath)
                compressed_size_kb = compressed_size_bytes / 1024
                size_reduction_kb = original_size_kb - compressed_size_kb
                is_disk_saved = size_reduction_kb > 0
                compression_percentage = (size_reduction_kb / original_size_kb) * 100 if original_size_kb > 0 else 0
                
                # Tính toán tỷ lệ giảm chiều ma trận lý thuyết của PCA
                # Số phần tử ma trận ban đầu: H * W
                # Số phần tử lưu trữ sau PCA: k * (H + W) + W
                orig_elements = H * W
                pca_elements = k * (H + W) + W
                theoretical_ratio = (pca_elements / orig_elements) * 100 if orig_elements > 0 else 0
                theoretical_savings = max(0.0, round(100.0 - theoretical_ratio, 1))
                
                compressed_size_display = f"{compressed_size_kb/1024:.2f} MB" if compressed_size_kb > 1024 else f"{compressed_size_kb:.2f} KB"
                size_reduction_display = f"{abs(size_reduction_kb)/1024:.2f} MB" if abs(size_reduction_kb) > 1024 else f"{abs(size_reduction_kb):.2f} KB"
                
                result = {
                    'mode': 'pca',
                    'orig': filename,
                    'recon': outname,
                    'k': k,
                    'raw_k': raw_k,
                    'max_k': max_k,
                    'img_width': W,
                    'img_height': H,
                    'adjusted_k': adjusted_k,
                    'mse': round(mse, 2),
                    'psnr': round(psnr, 2),
                    'time': round(elapsed, 2),
                    'original_size_kb': round(original_size_kb, 2),
                    'compressed_size_kb': round(compressed_size_kb, 2),
                    'original_size': original_size_display,
                    'compressed_size': compressed_size_display,
                    'size_reduction': size_reduction_display,
                    'is_disk_saved': is_disk_saved,
                    'compression_percentage': round(compression_percentage, 1),
                    'theoretical_savings': theoretical_savings,
                    'current_mode': 'pca',
                    'current_k': k
                }
                
            elif mode == 'enhance':
                enhance_method = request.form.get('enhance_method', 'combined')
                strength = int(request.form.get('strength', 2))
                
                start = time.time()
                orig_sharpness = calculate_sharpness(arr)
                enhanced_arr = enhance_image(arr, method=enhance_method, strength=strength)
                enhanced_sharpness = calculate_sharpness(enhanced_arr)
                elapsed = time.time() - start
                
                if orig_sharpness > 0:
                    sharpness_increase = ((enhanced_sharpness - orig_sharpness) / orig_sharpness) * 100
                else:
                    sharpness_increase = 0.0
                    
                outname = 'enhanced_' + filename
                outpath = os.path.join(app.config['UPLOAD_FOLDER'], outname)
                Image.fromarray(enhanced_arr).save(outpath, quality=95)
                
                enhanced_size_bytes = os.path.getsize(outpath)
                enhanced_size_kb = enhanced_size_bytes / 1024
                enhanced_size_display = f"{enhanced_size_kb/1024:.2f} MB" if enhanced_size_kb > 1024 else f"{enhanced_size_kb:.2f} KB"
                
                result = {
                    'mode': 'enhance',
                    'orig': filename,
                    'recon': outname,
                    'img_width': W,
                    'img_height': H,
                    'enhance_method': enhance_method,
                    'method_name': METHOD_NAMES.get(enhance_method, 'Làm rõ ảnh'),
                    'strength': strength,
                    'strength_label': STRENGTH_LABELS.get(strength, 'Vừa phải'),
                    'time': round(elapsed, 3),
                    'orig_sharpness': round(orig_sharpness, 1),
                    'enhanced_sharpness': round(enhanced_sharpness, 1),
                    'sharpness_increase': round(sharpness_increase, 1),
                    'original_size': original_size_display,
                    'compressed_size': enhanced_size_display,
                    'current_mode': 'enhance',
                    'current_method': enhance_method,
                    'current_strength': strength
                }
                
        except Exception as e:
            flash(f'Đã xảy ra lỗi trong quá trình xử lý: {str(e)}', 'danger')
            return render_template('index.html', result=None)

    return render_template('index.html', result=result)

@app.route('/enhance_existing', methods=['POST'])
def enhance_existing():
    filename = request.form.get('filename')
    enhance_method = request.form.get('enhance_method', 'combined')
    strength = int(request.form.get('strength', 2))
    
    if not filename:
        flash('Không tìm thấy ảnh để làm nét!', 'warning')
        return redirect(url_for('upload'))
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(filepath):
        flash('File ảnh không còn tồn tại trên server!', 'danger')
        return redirect(url_for('upload'))
        
    try:
        img = Image.open(filepath)
        img = convert_to_rgb(img)
        arr = np.array(img)
        
        H, W = arr.shape[0], arr.shape[1]
        original_size_bytes = os.path.getsize(filepath)
        original_size_kb = original_size_bytes / 1024
        original_size_display = f"{original_size_kb/1024:.2f} MB" if original_size_kb > 1024 else f"{original_size_kb:.2f} KB"
        
        start = time.time()
        orig_sharpness = calculate_sharpness(arr)
        enhanced_arr = enhance_image(arr, method=enhance_method, strength=strength)
        enhanced_sharpness = calculate_sharpness(enhanced_arr)
        elapsed = time.time() - start
        
        if orig_sharpness > 0:
            sharpness_increase = ((enhanced_sharpness - orig_sharpness) / orig_sharpness) * 100
        else:
            sharpness_increase = 0.0
            
        uid = uuid.uuid4().hex[:6]
        outname = f"enhanced_{uid}_{secure_filename(filename)}"
        outpath = os.path.join(app.config['UPLOAD_FOLDER'], outname)
        Image.fromarray(enhanced_arr).save(outpath, quality=95)
        
        enhanced_size_bytes = os.path.getsize(outpath)
        enhanced_size_kb = enhanced_size_bytes / 1024
        enhanced_size_display = f"{enhanced_size_kb/1024:.2f} MB" if enhanced_size_kb > 1024 else f"{enhanced_size_kb:.2f} KB"
        
        result = {
            'mode': 'enhance',
            'orig': secure_filename(filename),
            'recon': outname,
            'img_width': W,
            'img_height': H,
            'enhance_method': enhance_method,
            'method_name': METHOD_NAMES.get(enhance_method, 'Làm rõ ảnh'),
            'strength': strength,
            'strength_label': STRENGTH_LABELS.get(strength, 'Vừa phải'),
            'time': round(elapsed, 3),
            'orig_sharpness': round(orig_sharpness, 1),
            'enhanced_sharpness': round(enhanced_sharpness, 1),
            'sharpness_increase': round(sharpness_increase, 1),
            'original_size': original_size_display,
            'compressed_size': enhanced_size_display,
            'current_mode': 'enhance',
            'current_method': enhance_method,
            'current_strength': strength
        }
        return render_template('index.html', result=result)
        
    except Exception as e:
        flash(f'Lỗi khi làm nét ảnh: {str(e)}', 'danger')
        return redirect(url_for('upload'))

@app.route('/uploads/<name>')
def download(name):
    return send_from_directory(app.config['UPLOAD_FOLDER'], name)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
