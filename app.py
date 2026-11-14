# app.py
from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import os, io, time, uuid, glob
import cv2

app = Flask(__name__)
app.secret_key = 'pca_image_compression_secret_key_v2'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Giới hạn 32MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'jfif'}
ALLOWED_PACKAGE_EXTENSIONS = {'npz'}

def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_package_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_PACKAGE_EXTENSIONS

def cleanup_old_files(folder, max_age_seconds=7200):
    """Tự động dọn dẹp file tạm cũ hơn 2 giờ để tránh đầy ổ đĩa server"""
    try:
        now = time.time()
        for fname in os.listdir(folder):
            if fname.startswith('.') or fname == 'README.md':
                continue
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                if now - os.path.getmtime(fpath) > max_age_seconds:
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
    except Exception:
        pass

def convert_to_rgb(img):
    """Xử lý an toàn ảnh có kênh trong suốt (Alpha/RGBA) sang RGB nền trắng"""
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        bg = Image.new('RGB', rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return img.convert('RGB')

def calculate_sharpness(img_array):
    """Tính độ sắc nét dựa trên phương sai Laplacian (Laplacian variance)"""
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def calculate_ssim(img1, img2):
    """
    Tính chỉ số tương đồng cấu trúc (SSIM) theo chuẩn Zhou Wang
    giữa 2 ảnh numpy array uint8 (giá trị từ 0.0 đến 1.0)
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    if img1.ndim == 3:
        ssims = []
        for ch in range(img1.shape[2]):
            I1 = img1[:, :, ch].astype(np.float64)
            I2 = img2[:, :, ch].astype(np.float64)
            mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.GaussianBlur(I1 * I1, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(I2 * I2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(I1 * I2, (11, 11), 1.5) - mu1_mu2
            num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            ssims.append((num / den).mean())
        return float(np.clip(np.mean(ssims), 0.0, 1.0))
    else:
        I1 = img1.astype(np.float64)
        I2 = img2.astype(np.float64)
        mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.GaussianBlur(I1 * I1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(I2 * I2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(I1 * I2, (11, 11), 1.5) - mu1_mu2
        num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        return float(np.clip((num / den).mean(), 0.0, 1.0))

def compute_variance_profile(arr, max_eval=150):
    """
    Phân tích phổ phương sai (Scree Plot Profile) và tính ngưỡng k tối ưu
    cho 90%, 95%, 98%, 99% phương sai tích lũy
    """
    H, W = arr.shape[:2]
    max_k = min(H, W)
    eval_k = min(max_eval, max_k)
    
    if arr.ndim == 2:
        channels = [arr.astype(np.float64)]
    else:
        channels = [arr[:, :, ch].astype(np.float64) for ch in range(arr.shape[2])]
        
    channel_variances = []
    for ch in channels:
        centered = ch - ch.mean(axis=0)
        s = np.linalg.svd(centered, compute_uv=False)
        var = s ** 2
        total_var = np.sum(var)
        if total_var > 1e-10:
            channel_variances.append(var / total_var)
        else:
            channel_variances.append(np.zeros_like(var))
        
    mean_ratio = np.mean(channel_variances, axis=0)
    if np.sum(mean_ratio) > 1e-10:
        mean_ratio = mean_ratio / np.sum(mean_ratio)
        cum_ratio = np.cumsum(mean_ratio)
        k_90 = int(np.searchsorted(cum_ratio, 0.90) + 1)
        k_95 = int(np.searchsorted(cum_ratio, 0.95) + 1)
        k_98 = int(np.searchsorted(cum_ratio, 0.98) + 1)
        k_99 = int(np.searchsorted(cum_ratio, 0.99) + 1)
    else:
        cum_ratio = np.ones(max_k)
        mean_ratio = np.zeros(max_k)
        k_90 = k_95 = k_98 = k_99 = 1
    
    # Lấy các điểm đại diện để vẽ Scree Plot mượt mà trên Chart.js
    step = max(1, eval_k // 25)
    sample_ks = list(range(1, eval_k + 1, step))
    if eval_k not in sample_ks:
        sample_ks.append(eval_k)
        
    scree_data = {
        'ks': sample_ks,
        'cum_var': [round(float(cum_ratio[k - 1] * 100), 2) for k in sample_ks],
        'ind_var': [round(float(mean_ratio[k - 1] * 100), 2) for k in sample_ks]
    }
    
    return {
        'k_90': min(k_90, max_k),
        'k_95': min(k_95, max_k),
        'k_98': min(k_98, max_k),
        'k_99': min(k_99, max_k),
        'scree_data': scree_data,
        'cum_ratio': cum_ratio
    }

def run_pca_decomposition(arr, k):
    """
    Thực hiện phân rã PCA tối ưu hóa (sử dụng randomized SVD khi k nhỏ để tăng tốc)
    Tách biệt chi tiết thời gian phân tích và tái tạo (profiling).
    """
    H, W = arr.shape[0], arr.shape[1]
    max_k = min(H, W)
    k = max(1, min(k, max_k))
    
    svd_solver = 'randomized' if (k < int(0.75 * max_k) and max_k > 80) else 'auto'
    
    t_decomp_start = time.time()
    Z_list = []
    components_list = []
    mean_list = []
    
    if arr.ndim == 2:
        channel = arr.astype(np.float32)
        pca = PCA(n_components=k, svd_solver=svd_solver, random_state=42)
        comp = pca.fit_transform(channel)
        t_decomp = time.time() - t_decomp_start
        
        t_recon_start = time.time()
        recon = pca.inverse_transform(comp)
        recon = np.clip(recon, 0, 255).astype(np.uint8)
        t_recon = time.time() - t_recon_start
        
        Z_list.append(comp.astype(np.float16))
        components_list.append(pca.components_.astype(np.float16))
        mean_list.append(pca.mean_.astype(np.float16))
    else:
        recon = np.zeros_like(arr)
        t_recon_total = 0
        t_decomp_total = 0
        
        for ch in range(3):
            channel = arr[:, :, ch].astype(np.float32)
            t_ch_d0 = time.time()
            pca = PCA(n_components=k, svd_solver=svd_solver, random_state=42)
            comp = pca.fit_transform(channel)
            t_decomp_total += (time.time() - t_ch_d0)
            
            t_ch_r0 = time.time()
            ch_inv = pca.inverse_transform(comp)
            recon[:, :, ch] = np.clip(ch_inv, 0, 255)
            t_recon_total += (time.time() - t_ch_r0)
            
            Z_list.append(comp.astype(np.float16))
            components_list.append(pca.components_.astype(np.float16))
            mean_list.append(pca.mean_.astype(np.float16))
            
        recon = recon.astype(np.uint8)
        t_decomp = t_decomp_total
        t_recon = t_recon_total
        
    return {
        'recon': recon,
        'k': k,
        'Z_list': Z_list,
        'components_list': components_list,
        'mean_list': mean_list,
        't_decomp': t_decomp,
        't_recon': t_recon,
        'svd_solver': svd_solver
    }

def save_pca_package(filepath, Z_list, components_list, mean_list, orig_shape, k):
    """Lưu gói file nén PCA thực tế dạng nhị phân (.npz) có chứa ma trận Z, V, Mean"""
    np.savez_compressed(
        filepath,
        Z=np.stack(Z_list, axis=-1),
        V=np.stack(components_list, axis=0),
        mean=np.stack(mean_list, axis=0),
        orig_shape=np.array(orig_shape),
        k=k
    )
    return os.path.getsize(filepath)

def decompress_pca_package(filepath):
    """Giải nén và tái tạo ảnh từ gói nén PCA nhị phân (.npz)"""
    data = np.load(filepath)
    orig_shape = tuple(data['orig_shape'])
    k = int(data['k'])
    Z = data['Z'].astype(np.float32)
    V = data['V'].astype(np.float32)
    mean = data['mean'].astype(np.float32)
    
    t_start = time.time()
    if len(orig_shape) == 2 or orig_shape[2] == 1:
        comp = Z[:, :, 0] if Z.ndim == 3 else Z
        vk = V[0] if V.ndim == 3 else V
        m = mean[0] if mean.ndim == 2 else mean
        recon = np.dot(comp, vk) + m
        recon = np.clip(recon, 0, 255).astype(np.uint8)
    else:
        recon = np.zeros(orig_shape, dtype=np.uint8)
        for ch in range(3):
            comp = Z[:, :, ch]
            vk = V[ch]
            m = mean[ch]
            ch_recon = np.dot(comp, vk) + m
            recon[:, :, ch] = np.clip(ch_recon, 0, 255)
            
    elapsed = time.time() - t_start
    return recon, k, orig_shape, elapsed

def enhance_image(img_array, method='combined', strength=2, denoise=True):
    """
    Làm rõ / tăng nét ảnh bằng OpenCV & Pillow
    - img_array: numpy array RGB uint8
    - method: 'combined', 'unsharp', 'clahe', 'laplacian'
    - strength: 1, 2, 3, 4, 5
    - denoise: Khử nhiễu lọc song phương (Bilateral Filter) trước khi làm nét
    """
    strength_map = {1: 0.8, 2: 1.4, 3: 2.0, 4: 2.8, 5: 3.8}
    weight = strength_map.get(strength, 1.4)
    img = img_array.copy()
    
    # Tiền xử lý khử nhiễu nhẹ nếu được kích hoạt (bảo toàn viền cạnh)
    if denoise:
        img = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)
    
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

@app.route('/', methods=['GET', 'POST'])
def upload():
    # Tự động dọn file cũ
    cleanup_old_files(app.config['UPLOAD_FOLDER'])
    
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
            uid = uuid.uuid4().hex[:8]
            safe_name = secure_filename(file.filename)
            filename = f"{uid}_{safe_name}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            original_size_bytes = os.path.getsize(filepath)
            original_size_kb = original_size_bytes / 1024
            original_size_display = f"{original_size_kb/1024:.2f} MB" if original_size_kb > 1024 else f"{original_size_kb:.2f} KB"
            
            t_prep0 = time.time()
            img = Image.open(filepath)
            img = convert_to_rgb(img)
            arr = np.array(img)
            t_preprocess = time.time() - t_prep0
            
            H, W = arr.shape[0], arr.shape[1]
            max_k = min(H, W)
            
            if mode == 'pca':
                raw_k = int(request.form.get('k', 50))
                adjusted_k = False
                if raw_k > max_k:
                    k = max_k
                    adjusted_k = True
                elif raw_k < 1:
                    k = 1
                else:
                    k = raw_k
                    
                # Phân tích phương sai tích lũy và Scree plot
                var_profile = compute_variance_profile(arr, max_eval=min(150, max_k))
                cum_energy = float(var_profile['cum_ratio'][k - 1] * 100) if (k - 1) < len(var_profile['cum_ratio']) else 100.0
                
                # Thực hiện phân rã PCA
                pca_res = run_pca_decomposition(arr, k)
                recon = pca_res['recon']
                
                # Tính MSE, PSNR, SSIM
                orig_f = arr.astype(float)
                rec_f = recon.astype(float)
                mse = np.mean((orig_f - rec_f) ** 2)
                psnr = 10 * np.log10((255.0 ** 2) / mse) if mse > 0 else float('inf')
                ssim_val = calculate_ssim(arr, recon)
                
                # Lưu ảnh tái tạo JPG
                outname = 'recon_' + filename
                outpath = os.path.join(app.config['UPLOAD_FOLDER'], outname)
                Image.fromarray(recon).save(outpath, quality=92)
                
                # Lưu gói nén PCA thực tế (.npz)
                pkg_name = f"pca_pkg_{uid}_{os.path.splitext(safe_name)[0]}.npz"
                pkg_path = os.path.join(app.config['UPLOAD_FOLDER'], pkg_name)
                npz_size_bytes = save_pca_package(
                    pkg_path, pca_res['Z_list'], pca_res['components_list'], 
                    pca_res['mean_list'], arr.shape, k
                )
                npz_size_kb = npz_size_bytes / 1024
                npz_savings_pct = max(0.0, (original_size_bytes - npz_size_bytes) / original_size_bytes * 100) if original_size_bytes > 0 else 0
                
                # Kích thước file thực tế trên ổ đĩa
                compressed_size_bytes = os.path.getsize(outpath)
                compressed_size_kb = compressed_size_bytes / 1024
                size_reduction_kb = original_size_kb - compressed_size_kb
                is_disk_saved = size_reduction_kb > 0
                compression_percentage = (size_reduction_kb / original_size_kb) * 100 if original_size_kb > 0 else 0
                
                # Giảm chiều ma trận lý thuyết
                orig_elements = H * W
                pca_elements = k * (H + W) + W
                theoretical_ratio = (pca_elements / orig_elements) * 100 if orig_elements > 0 else 0
                theoretical_savings = max(0.0, round(100.0 - theoretical_ratio, 1))
                
                compressed_size_display = f"{compressed_size_kb/1024:.2f} MB" if compressed_size_kb > 1024 else f"{compressed_size_kb:.2f} KB"
                size_reduction_display = f"{abs(size_reduction_kb)/1024:.2f} MB" if abs(size_reduction_kb) > 1024 else f"{abs(size_reduction_kb):.2f} KB"
                npz_size_display = f"{npz_size_kb/1024:.2f} MB" if npz_size_kb > 1024 else f"{npz_size_kb:.2f} KB"
                
                total_elapsed = t_preprocess + pca_res['t_decomp'] + pca_res['t_recon']
                
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
                    'ssim': round(ssim_val, 4),
                    'ssim_percent': round(ssim_val * 100, 1),
                    'cum_energy': round(cum_energy, 2),
                    'time': round(total_elapsed, 3),
                    't_preprocess': round(t_preprocess, 3),
                    't_decomp': round(pca_res['t_decomp'], 3),
                    't_recon': round(pca_res['t_recon'], 3),
                    'svd_solver': pca_res['svd_solver'],
                    'original_size_kb': round(original_size_kb, 2),
                    'compressed_size_kb': round(compressed_size_kb, 2),
                    'original_size': original_size_display,
                    'compressed_size': compressed_size_display,
                    'size_reduction': size_reduction_display,
                    'is_disk_saved': is_disk_saved,
                    'compression_percentage': round(compression_percentage, 1),
                    'theoretical_savings': theoretical_savings,
                    'current_mode': 'pca',
                    'current_k': k,
                    # NPZ Real compression package
                    'pkg_name': pkg_name,
                    'npz_size': npz_size_display,
                    'npz_size_kb': round(npz_size_kb, 1),
                    'npz_savings_pct': round(npz_savings_pct, 1),
                    # Auto k suggestions & Scree Plot
                    'k_90': var_profile['k_90'],
                    'k_95': var_profile['k_95'],
                    'k_98': var_profile['k_98'],
                    'k_99': var_profile['k_99'],
                    'scree_ks': var_profile['scree_data']['ks'],
                    'scree_cum_var': var_profile['scree_data']['cum_var'],
                    'scree_ind_var': var_profile['scree_data']['ind_var']
                }
                
            elif mode == 'enhance':
                enhance_method = request.form.get('enhance_method', 'combined')
                strength = int(request.form.get('strength', 2))
                denoise = request.form.get('denoise', '1') == '1'
                
                start = time.time()
                orig_sharpness = calculate_sharpness(arr)
                enhanced_arr = enhance_image(arr, method=enhance_method, strength=strength, denoise=denoise)
                enhanced_sharpness = calculate_sharpness(enhanced_arr)
                elapsed = time.time() - start
                
                sharpness_increase = ((enhanced_sharpness - orig_sharpness) / orig_sharpness) * 100 if orig_sharpness > 0 else 0.0
                ssim_enhanced = calculate_ssim(arr, enhanced_arr)
                    
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
                    'denoise': denoise,
                    'time': round(elapsed, 3),
                    'orig_sharpness': round(orig_sharpness, 1),
                    'enhanced_sharpness': round(enhanced_sharpness, 1),
                    'sharpness_increase': round(sharpness_increase, 1),
                    'ssim': round(ssim_enhanced, 4),
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

@app.route('/sweep_pca', methods=['POST'])
def sweep_pca():
    """Chế độ so sánh nhanh 4 mức k khác nhau (Multi-k Sweep) trên cùng ảnh"""
    filename = request.form.get('filename')
    if not filename:
        return jsonify({'error': 'Không tìm thấy file'}), 400
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify({'error': 'File không tồn tại trên server'}), 404
        
    try:
        img = Image.open(filepath)
        img = convert_to_rgb(img)
        arr = np.array(img)
        H, W = arr.shape[:2]
        max_k = min(H, W)
        
        # Lấy 4 mức k theo tỷ lệ hợp lý
        var_profile = compute_variance_profile(arr, max_eval=min(120, max_k))
        k1 = max(5, int(var_profile['k_90'] * 0.4))
        k2 = var_profile['k_90']
        k3 = var_profile['k_95']
        k4 = var_profile['k_99']
        
        sweep_ks = sorted(list(set([k1, k2, k3, k4])))
        results = []
        
        for cur_k in sweep_ks:
            pca_res = run_pca_decomposition(arr, cur_k)
            recon = pca_res['recon']
            orig_f = arr.astype(float)
            rec_f = recon.astype(float)
            mse = np.mean((orig_f - rec_f) ** 2)
            psnr = 10 * np.log10((255.0 ** 2) / mse) if mse > 0 else 99.0
            ssim_val = calculate_ssim(arr, recon)
            
            sweep_outname = f"sweep_k{cur_k}_{secure_filename(filename)}"
            sweep_outpath = os.path.join(app.config['UPLOAD_FOLDER'], sweep_outname)
            Image.fromarray(recon).save(sweep_outpath, quality=88)
            
            # Tính phần trăm ma trận tiết kiệm
            orig_elements = H * W
            pca_elements = cur_k * (H + W) + W
            savings = max(0.0, round((1.0 - (pca_elements / orig_elements)) * 100, 1)) if orig_elements > 0 else 0
            
            cum_energy = float(var_profile['cum_ratio'][cur_k - 1] * 100) if (cur_k - 1) < len(var_profile['cum_ratio']) else 100.0
            
            results.append({
                'k': cur_k,
                'image_url': f"/uploads/{sweep_outname}",
                'mse': round(mse, 2),
                'psnr': round(psnr, 2),
                'ssim': round(ssim_val, 4),
                'energy': round(cum_energy, 1),
                'savings': savings
            })
            
        return jsonify({'success': True, 'sweep_results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/decompress_pca', methods=['POST'])
def decompress_pca():
    """Tải lên file gói nén .npz và giải nén tái tạo lại ảnh gốc"""
    file = request.files.get('package_file')
    if not file or file.filename == '':
        flash('Vui lòng chọn file gói nén .npz!', 'warning')
        return redirect(url_for('upload'))
        
    if not allowed_package_file(file.filename):
        flash('Chỉ hỗ trợ file gói nén định dạng .npz!', 'danger')
        return redirect(url_for('upload'))
        
    try:
        uid = uuid.uuid4().hex[:8]
        safe_name = secure_filename(file.filename)
        pkg_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_{safe_name}")
        file.save(pkg_path)
        
        recon, k, orig_shape, elapsed = decompress_pca_package(pkg_path)
        
        outname = f"decompressed_{uid}.png"
        outpath = os.path.join(app.config['UPLOAD_FOLDER'], outname)
        Image.fromarray(recon).save(outpath)
        
        pkg_size_kb = os.path.getsize(pkg_path) / 1024
        recon_size_kb = os.path.getsize(outpath) / 1024
        
        result = {
            'mode': 'pca',
            'orig': outname,
            'recon': outname,
            'k': k,
            'raw_k': k,
            'max_k': min(orig_shape[0], orig_shape[1]),
            'img_width': orig_shape[1],
            'img_height': orig_shape[0],
            'adjusted_k': False,
            'mse': 0.0,
            'psnr': 99.0,
            'ssim': 1.0,
            'ssim_percent': 100.0,
            'cum_energy': 100.0,
            'time': round(elapsed, 4),
            't_preprocess': 0.0,
            't_decomp': 0.0,
            't_recon': round(elapsed, 4),
            'svd_solver': 'NPZ Unpack',
            'original_size_kb': round(pkg_size_kb, 2),
            'compressed_size_kb': round(recon_size_kb, 2),
            'original_size': f"{pkg_size_kb:.2f} KB (File nén .npz)",
            'compressed_size': f"{recon_size_kb:.2f} KB (Sau giải nén)",
            'size_reduction': "0 KB",
            'is_disk_saved': True,
            'compression_percentage': 0,
            'theoretical_savings': 0,
            'current_mode': 'pca',
            'current_k': k,
            'is_decompressed_view': True,
            'k_90': k, 'k_95': k, 'k_98': k, 'k_99': k,
            'scree_ks': [k], 'scree_cum_var': [100.0], 'scree_ind_var': [100.0]
        }
        flash(f'Giải nén thành công từ file .npz ({orig_shape[1]}x{orig_shape[0]} px, k={k}) trong {elapsed:.3f}s!', 'success')
        return render_template('index.html', result=result)
        
    except Exception as e:
        flash(f'Lỗi khi giải nén file .npz: {str(e)}', 'danger')
        return redirect(url_for('upload'))

@app.route('/download_npz/<filename>')
def download_npz(filename):
    """Tải xuống file gói nén PCA (.npz)"""
    safe_name = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_name, as_attachment=True)

@app.route('/enhance_existing', methods=['POST'])
def enhance_existing():
    filename = request.form.get('filename')
    enhance_method = request.form.get('enhance_method', 'combined')
    strength = int(request.form.get('strength', 2))
    denoise = request.form.get('denoise', '1') == '1'
    
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
        enhanced_arr = enhance_image(arr, method=enhance_method, strength=strength, denoise=denoise)
        enhanced_sharpness = calculate_sharpness(enhanced_arr)
        elapsed = time.time() - start
        
        sharpness_increase = ((enhanced_sharpness - orig_sharpness) / orig_sharpness) * 100 if orig_sharpness > 0 else 0.0
        ssim_val = calculate_ssim(arr, enhanced_arr)
            
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
            'denoise': denoise,
            'time': round(elapsed, 3),
            'orig_sharpness': round(orig_sharpness, 1),
            'enhanced_sharpness': round(enhanced_sharpness, 1),
            'sharpness_increase': round(sharpness_increase, 1),
            'ssim': round(ssim_val, 4),
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
    return send_from_directory(app.config['UPLOAD_FOLDER'], secure_filename(name))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
