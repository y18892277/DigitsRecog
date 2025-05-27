import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, N, S, E, W
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os
import cv2

# --- 全局变量和常量 ---
IMAGE_DISPLAY_SIZE = 280 # 预览区域显示图片的最大尺寸
IMAGE_SIZE = 28 # MNIST图像大小
DEBUG_IMAGE_DIR = "debug_images"

# --- 创建调试文件夹 ---
if not os.path.exists(DEBUG_IMAGE_DIR):
    try:
        os.makedirs(DEBUG_IMAGE_DIR)
        print(f"调试文件夹 '{DEBUG_IMAGE_DIR}' 已创建。")
    except Exception as e_dir:
        print(f"创建调试文件夹 '{DEBUG_IMAGE_DIR}' 失败: {e_dir}")
        DEBUG_IMAGE_DIR = "." # 如果创建失败，则保存在当前目录

# --- 模型加载 ---
try:
    model = tf.keras.models.load_model('model.h5')
    print("模型 model.h5 加载成功！(image_uploader_recognizer.py)")
except Exception as e:
    print(f"模型加载失败 (image_uploader_recognizer.py): {e}")
    model = None

class ImageRecognizerApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("图片手写数字识别 (OpenCV)")

        main_app_frame = Frame(self.root)
        main_app_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.left_frame = Frame(main_app_frame)
        self.left_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.BOTH, expand=True)

        Label(self.left_frame, text="原始图片预览:", font=("Arial", 12)).pack(pady=(0,5))
        self.image_frame = Frame(self.left_frame, bd=2, relief=tk.SUNKEN, 
                                 width=IMAGE_DISPLAY_SIZE, height=IMAGE_DISPLAY_SIZE)
        self.image_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        self.image_frame.pack_propagate(False) 
        self.image_label = Label(self.image_frame, bg="lightgray")
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
        self.right_frame = Frame(main_app_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        Label(self.right_frame, text="预处理后 (送入模型): (28x28 放大显示)", justify=tk.LEFT, font=("Arial", 10)).pack(pady=(0,5))
        self.processed_image_display_size = IMAGE_SIZE * 6 
        self.processed_image_frame = Frame(self.right_frame, bd=2, relief=tk.SUNKEN,
                                           width=self.processed_image_display_size, 
                                           height=self.processed_image_display_size)
        self.processed_image_frame.pack(pady=5, fill=tk.NONE, expand=False)
        self.processed_image_frame.pack_propagate(False)
        self.processed_image_label = Label(self.processed_image_frame, bg="darkgray")
        self.processed_image_label.pack(expand=True, fill=tk.BOTH)

        self.controls_frame = Frame(self.right_frame)
        self.controls_frame.pack(pady=(20,0), fill=tk.X)
        
        self.upload_button = Button(self.controls_frame, text="上传图片并识别", 
                                    command=self.upload_and_predict, width=20, height=2)
        self.upload_button.grid(row=0, column=0, padx=5, pady=5, sticky=W+E)

        self.result_display_frame = Frame(self.controls_frame)
        self.result_display_frame.grid(row=0, column=1, padx=5, pady=5, sticky=W+E)

        Label(self.result_display_frame, text="识别结果:", font=("Arial", 14)).pack(side=tk.LEFT, anchor=S, padx=(5,0))
        self.result_label = Label(self.result_display_frame, text="?", font=("Arial", 40, "bold"), 
                                  fg="blue", width=2, relief=tk.RIDGE)
        self.result_label.pack(side=tk.LEFT, anchor=S, padx=5)
        
        self.controls_frame.grid_columnconfigure(0, weight=1)
        self.controls_frame.grid_columnconfigure(1, weight=1)

    def clear_processed_preview(self):
        if hasattr(self, 'processed_image_label'):
            self.processed_image_label.config(image='')
            self.processed_image_label.image = None

    def upload_and_predict(self):
        if model is None:
            self.result_label.config(text="ERR")
            print("模型未加载，无法识别。")
            if hasattr(self, 'image_label'):
                 self.image_label.config(image='')
                 self.image_label.image = None 
            self.clear_processed_preview()
            return

        filepath = filedialog.askopenfilename(
            title="选择一个手写数字图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"), 
                       ("所有文件", "*.*")]
        )
        if not filepath: 
            return
        
        self.clear_processed_preview()
        self.result_label.config(text="...")
        self.root.update_idletasks()

        try:
            pil_image_initial = Image.open(filepath)
            try:
                pil_image_initial.save(os.path.join(DEBUG_IMAGE_DIR, "0_original_uploaded_pil.png"))
            except Exception as e_save:
                print(f"保存 0_original_uploaded_pil.png 失败: {e_save}")

            open_cv_image_original_color = np.array(pil_image_initial.convert('RGB'))
            open_cv_image_original_color = open_cv_image_original_color[:, :, ::-1].copy() # RGB to BGR
            cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, "0a_original_opencv_bgr.png"), open_cv_image_original_color)

            display_image_tk = pil_image_initial.copy()
            display_image_tk.thumbnail((IMAGE_DISPLAY_SIZE, IMAGE_DISPLAY_SIZE), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(display_image_tk)
            self.image_label.config(image=self.tk_image)
            self.image_label.image = self.tk_image
            
            max_dim_prescale = 400
            h_orig, w_orig = open_cv_image_original_color.shape[:2]
            pre_scaled_cv_image = open_cv_image_original_color.copy()
            if h_orig > max_dim_prescale or w_orig > max_dim_prescale:
                if h_orig > w_orig:
                    new_h = max_dim_prescale
                    new_w = int(w_orig * (new_h / h_orig))
                else:
                    new_w = max_dim_prescale
                    new_h = int(h_orig * (new_w / w_orig))
                pre_scaled_cv_image = cv2.resize(open_cv_image_original_color, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, "1_prescaled_cv.png"), pre_scaled_cv_image)

            gray_cv = cv2.cvtColor(pre_scaled_cv_image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, "2_gray_cv.png"), gray_cv)

            # 重新启用高斯模糊 (3,3)
            blurred_cv = cv2.GaussianBlur(gray_cv, (3, 3), 0) 
            cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, "3_blurred_cv_k3_for_adaptive.png"), blurred_cv)

            # 使用自适应阈值
            blockSize = 11
            C_val = 2
            binary_cv_hdbz = cv2.adaptiveThreshold(blurred_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                  cv2.THRESH_BINARY_INV, blockSize, C_val)
            cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"4_adaptive_gaussian_inv_b{blockSize}_c{C_val}_HDBZ.png"), binary_cv_hdbz)
            
            kernel_size_morph = (3,3) 
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size_morph)
            closed_cv_hdbz = cv2.morphologyEx(binary_cv_hdbz, cv2.MORPH_CLOSE, kernel, iterations=1)
            cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"5a_morph_close_b{blockSize}_c{C_val}_k{kernel_size_morph[0]}.png"), closed_cv_hdbz)
            
            # 新增：进行开运算以清理背景噪点 (在之前的闭运算之后)
            # 使用与闭运算相同的核大小作为初始尝试，可以根据效果调整
            kernel_open_size = (2,2) # 原为 kernel_size_morph (3,3)，调整为 (2,2) 以减少对数字线条的破坏
            kernel_for_opening = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_open_size)
            # iterations=1 通常足够，也可调整
            opened_after_close_cv = cv2.morphologyEx(closed_cv_hdbz, cv2.MORPH_OPEN, kernel_for_opening, iterations=1)
            cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"5b_opened_after_close_b{blockSize}_c{C_val}_k{kernel_open_size[0]}.png"), opened_after_close_cv)

            # 新增步骤：通过轮廓面积进一步清理背景白色小噪点
            image_for_noise_filtering = opened_after_close_cv.copy()
            # 使用 RETR_LIST 获取所有轮廓，包括嵌套的，以确保捕捉所有潜在的小噪点
            # 使用 CHAIN_APPROX_SIMPLE 压缩轮廓点，节省内存，对面积计算无影响
            noise_contours, _ = cv2.findContours(image_for_noise_filtering, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            max_noise_area_threshold = 20 # 定义噪点的最大面积阈值，小于此面积的白色区域将被移除。用户之前改为15导致数字断裂，现调整为5以保护数字完整性
            num_noise_removed = 0
            if noise_contours:
                for c in noise_contours:
                    area = cv2.contourArea(c)
                    if area < max_noise_area_threshold and area > 0: # 面积大于0确保是有效轮廓
                        cv2.drawContours(image_for_noise_filtering, [c], -1, 0, -1)
                        num_noise_removed += 1
            print(f"从5b图像中移除了 {num_noise_removed} 个面积小于 {max_noise_area_threshold} 的噪点。")
            cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"5c_contour_filtered_b{blockSize}_c{C_val}_k{kernel_open_size[0]}_T{max_noise_area_threshold}.png"), image_for_noise_filtering)
            
            # 新增步骤5d：从5c的结果中仅保留最大的轮廓（假定为数字）
            # 我们需要复制 image_for_noise_filtering 因为 findContours 会修改源图像
            contours_in_5c, _ = cv2.findContours(image_for_noise_filtering.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_with_only_digit = np.zeros_like(image_for_noise_filtering) # 创建一个同样大小的黑色背景图像
            
            if contours_in_5c:
                if len(contours_in_5c) > 0:
                    largest_contour_from_5c = max(contours_in_5c, key=cv2.contourArea)
                    # 新方法：使用位运算保留最大轮廓的内部结构（如孔洞）
                    # 1. 创建一个只包含最大轮廓（实心）的掩码
                    mask_for_largest = np.zeros_like(image_for_noise_filtering)
                    cv2.drawContours(mask_for_largest, [largest_contour_from_5c], -1, 255, -1) # -1表示填充
                    
                    # 2. 使用此掩码从 image_for_noise_filtering (5c的输出) 中提取原始像素
                    image_with_only_digit = cv2.bitwise_and(image_for_noise_filtering, image_for_noise_filtering, mask=mask_for_largest)
                    print(f"步骤5d：已从5c的输出中分离出最大轮廓（保留内部结构）。")
                else:
                    print(f"步骤5d警告：在5c的输出中未找到轮廓可供分离（contours_in_5c列表为空）。")    
            else:
                print(f"步骤5d警告：在5c的输出中未找到轮廓可供分离（contours_in_5c为None）。")

            cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"5d_isolated_digit_from_5c_b{blockSize}_c{C_val}_k{kernel_open_size[0]}_T{max_noise_area_threshold}.png"), image_with_only_digit)
            processed_binary_for_contours = image_with_only_digit # 更新后续轮廓检测的输入

            contours, _ = cv2.findContours(processed_binary_for_contours.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            final_mnist_like_image_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) 

            if contours:
                contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)
                digit_roi_cv = processed_binary_for_contours[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"6_cropped_roi_b{blockSize}_c{C_val}.png"), digit_roi_cv)
                
                # 新增：对裁剪出的ROI进行一次额外的开运算，以清理附着在数字上的小噪点
                # 使用一个较小的核，例如 (2,2)，以避免过度侵蚀数字本身
                kernel_roi_open_size = (3,3)
                kernel_for_roi_opening = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_roi_open_size)
                # 确保 digit_roi_cv 不为空，并且是二值图像 (虽然它应该是)
                if digit_roi_cv.size > 0 and digit_roi_cv.ndim == 2:
                    cleaned_digit_roi_cv = cv2.morphologyEx(digit_roi_cv, cv2.MORPH_OPEN, kernel_for_roi_opening, iterations=1)
                    cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"6a_roi_opened_b{blockSize}_c{C_val}_k{kernel_roi_open_size[0]}.png"), cleaned_digit_roi_cv)
                else:
                    print(f"警告:裁剪后的ROI为空或格式不正确，跳过ROI开运算。ROI shape: {digit_roi_cv.shape}")
                    cleaned_digit_roi_cv = digit_roi_cv # 如果有问题，则使用原始ROI

                padding = 4 
                max_content_dim = IMAGE_SIZE - 2 * padding
                
                roi_h, roi_w = cleaned_digit_roi_cv.shape[:2] # 使用清理后的ROI的尺寸
                if roi_w == 0 or roi_h == 0:
                    print("警告: 清理后的ROI尺寸为零。")
                else:
                    if roi_w > roi_h:
                        new_w_roi = max_content_dim
                        new_h_roi = int(roi_h * (new_w_roi / roi_w))
                    else:
                        new_h_roi = max_content_dim
                        new_w_roi = int(roi_w * (new_h_roi / roi_h))
                    
                    new_w_roi = max(1, new_w_roi)
                    new_h_roi = max(1, new_h_roi)

                    scaled_digit_roi_cv = cv2.resize(cleaned_digit_roi_cv, (new_w_roi, new_h_roi), interpolation=cv2.INTER_LINEAR) # 使用清理后的ROI进行缩放
                    cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"7_scaled_roi_b{blockSize}_c{C_val}_INTER_LINEAR.png"), scaled_digit_roi_cv)

                    # 新增：对缩放后的ROI进行闭运算，以修补可能的断裂
                    # scaled_digit_roi_cv 通常是 0-255 的灰度图，但其内容主要是二值的
                    # 为了确保形态学操作效果，可以先将其转为严格的二值图像 (0 或 255)
                    # 不过，如果其已经是近似二值（例如，背景是0，前景是255），直接操作通常也可以
                    # 这里我们假设它可以直接用于形态学操作，如果效果不佳，再考虑强制二值化
                    repair_kernel_size = (2,2) # 使用较小的核进行修补
                    kernel_for_repair = cv2.getStructuringElement(cv2.MORPH_RECT, repair_kernel_size)
                    
                    # 确保 scaled_digit_roi_cv 不为空且为2D图像
                    if scaled_digit_roi_cv.size > 0 and scaled_digit_roi_cv.ndim == 2:
                        repaired_scaled_roi_cv = cv2.morphologyEx(scaled_digit_roi_cv, cv2.MORPH_CLOSE, kernel_for_repair, iterations=1)
                        cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"7a_repaired_scaled_roi_b{blockSize}_c{C_val}_k{repair_kernel_size[0]}.png"), repaired_scaled_roi_cv)
                    else:
                        print(f"警告: 缩放后的ROI为空或格式不正确，跳过ROI修复。ROI shape: {scaled_digit_roi_cv.shape}")
                        repaired_scaled_roi_cv = scaled_digit_roi_cv # 如果有问题，则使用原始缩放ROI

                    paste_x = (IMAGE_SIZE - new_w_roi) // 2
                    paste_y = (IMAGE_SIZE - new_h_roi) // 2
                    
                    final_mnist_like_image_np[paste_y : paste_y + new_h_roi, paste_x : paste_x + new_w_roi] = repaired_scaled_roi_cv / 255.0 # 使用修复后的ROI
            else:
                print("警告: OpenCV未能找到任何轮廓。")

            Image.fromarray((final_mnist_like_image_np * 255).astype(np.uint8)).save(os.path.join(DEBUG_IMAGE_DIR, f"9_final_for_model_b{blockSize}_c{C_val}.png"))

            try:
                processed_display_array_gui = (final_mnist_like_image_np * 255).astype(np.uint8)
                self.processed_pil_to_show = Image.fromarray(processed_display_array_gui, mode='L')
                display_scale_factor = self.processed_image_display_size // IMAGE_SIZE
                scaled_width_gui = IMAGE_SIZE * display_scale_factor
                scaled_height_gui = IMAGE_SIZE * display_scale_factor
                self.processed_pil_to_show_resized = self.processed_pil_to_show.resize(
                    (scaled_width_gui, scaled_height_gui), Image.Resampling.NEAREST
                )
                self.processed_tk_image_to_show = ImageTk.PhotoImage(self.processed_pil_to_show_resized)
                self.processed_image_label.config(image=self.processed_tk_image_to_show)
                self.processed_image_label.image = self.processed_tk_image_to_show
            except Exception as e_display_processed:
                print(f"显示OpenCV预处理后图像时出错: {e_display_processed}")
                self.clear_processed_preview()
            
            img_to_predict = final_mnist_like_image_np.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

            prediction = model.predict(img_to_predict)
            predicted_digit = np.argmax(prediction)
            self.result_label.config(text=str(predicted_digit))
            print(f"图片 '{filepath}' (OpenCV b{blockSize}_c{C_val})预测结果: {predicted_digit}, 原始预测向量: {prediction}")

        except FileNotFoundError:
            self.result_label.config(text="ERR")
            if hasattr(self, 'image_label'): self.image_label.config(image=''); self.image_label.image = None
            self.clear_processed_preview()
            print(f"错误: 文件未找到 {filepath}")
        except Exception as e:
            self.result_label.config(text="ERR")
            if hasattr(self, 'image_label'): self.image_label.config(image=''); self.image_label.image = None
            self.clear_processed_preview()
            print(f"处理图片或识别过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRecognizerApp(root)
    root.mainloop()
