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

            # 新增步骤4b：通过轮廓面积分析去除步骤4输出中的微小白色噪点
            image_for_4b_filtering = binary_cv_hdbz.copy()
            contours_step4, _ = cv2.findContours(image_for_4b_filtering, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            max_white_noise_area_after_4 = 50 # 定义噪点的最大面积阈值 (白色区域)
            num_noise_removed_4b = 0
            if contours_step4:
                for c in contours_step4:
                    area = cv2.contourArea(c)
                    if area < max_white_noise_area_after_4 and area > 0: # 确保是有效的小面积轮廓
                        # 在副本上用黑色填充掉这些小轮廓
                        cv2.drawContours(image_for_4b_filtering, [c], -1, 0, -1) 
                        num_noise_removed_4b += 1
            
            if num_noise_removed_4b > 0:
                print(f"步骤4b：从步骤4的输出中移除了 {num_noise_removed_4b} 个面积小于 {max_white_noise_area_after_4} 的白色噪点区域。")
                binary_cv_hdbz = image_for_4b_filtering # 更新主图像变量
                cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"4b_white_noise_filtered_T{max_white_noise_area_after_4}_after_4.png"), binary_cv_hdbz)
            else:
                print(f"步骤4b：在步骤4的输出中未找到或未移除任何面积小于 {max_white_noise_area_after_4} 的白色噪点区域。")
                # 如果没有移除任何东西，可以选择不保存4b的图像，或者照常保存（它将和4一样）
                # 为了调试明确，即使没有改动也保存，可以看出此步骤被执行了
                cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"4b_white_noise_filtered_T{max_white_noise_area_after_4}_after_4_no_change.png"), binary_cv_hdbz)

            # 原步骤5a (闭运算) 将作用于经过4b处理（或未处理）的binary_cv_hdbz
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
            
            # --- 开始新的多数字识别逻辑 (基于5c的输出) ---
            # image_for_noise_filtering (5c的输出) 是我们寻找多个轮廓的基础
            all_potential_digit_contours, _ = cv2.findContours(image_for_noise_filtering.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            recognized_digits_list = []
            debug_candidates_display_img = None # 用于存储带有候选框的预览图像 (M5e)

            if all_potential_digit_contours and len(all_potential_digit_contours) > 0:
                min_digit_area = 25  # 定义一个轮廓被视为可能数字的最小面积
                
                candidate_contours = []
                for c_contour in all_potential_digit_contours:
                    area = cv2.contourArea(c_contour)
                    if area >= min_digit_area:
                        candidate_contours.append(c_contour)
                
                if not candidate_contours:
                    print("多数字识别：在5c之后未找到符合面积条件的候选数字轮廓。")
                    self.result_label.config(text="?")
                    self.clear_processed_preview()
                else:
                    # 按x坐标对候选轮廓排序（从左到右）
                    candidate_contours.sort(key=lambda c_sort: cv2.boundingRect(c_sort)[0])
                    
                    # 创建一个彩色图像用于绘制候选框 (M5e)
                    debug_candidates_display_img = cv2.cvtColor(image_for_noise_filtering.copy(), cv2.COLOR_GRAY2BGR)
                    
                    digit_index = 0
                    for contour_item in candidate_contours:
                        # 为每个数字重置28x28的画布
                        final_mnist_like_image_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) 

                        x, y, w, h = cv2.boundingRect(contour_item)
                        
                        # 在 debug_candidates_display_img 上绘制绿色候选框和索引
                        cv2.rectangle(debug_candidates_display_img, (x, y), (x + w, y + h), (0, 255, 0), 1) 
                        cv2.putText(debug_candidates_display_img, str(digit_index), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)

                        # M6: 从 image_for_noise_filtering (5c的输出) 裁剪ROI
                        digit_roi_cv = image_for_noise_filtering[y:y+h, x:x+w]
                        cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"M6_cropped_roi_digit{digit_index}_b{blockSize}_c{C_val}_T{max_noise_area_threshold}.png"), digit_roi_cv)
                        
                        # M6a步骤已被用户要求取消
                        # 原M6a: 对裁剪出的ROI进行开运算清理 (使用(3,3)核)
                        cleaned_digit_roi_cv = digit_roi_cv # 直接使用M6裁剪的ROI作为cleaned_digit_roi_cv
                        
                        # 原先的M6a的else条件和空ROI检查仍然有用，但现在是针对digit_roi_cv
                        if not (digit_roi_cv.size > 0 and digit_roi_cv.ndim == 2):
                            print(f"警告: 数字 {digit_index} 的裁剪ROI (M6的输出) 为空或格式不正确。 ROI shape: {digit_roi_cv.shape}")
                            if digit_roi_cv.size == 0: 
                                print(f"数字 {digit_index} ROI为空，无法处理，跳过。")
                                digit_index +=1
                                continue
                            # 如果格式不正确但非空，后续步骤可能会出错，但流程会继续，cleaned_digit_roi_cv 保持为 digit_roi_cv

                        # M7: 缩放ROI (现在输入是 cleaned_digit_roi_cv，即M6的输出)
                        padding = 4 
                        max_content_dim = IMAGE_SIZE - 2 * padding
                        roi_h, roi_w = cleaned_digit_roi_cv.shape[:2]

                        if roi_w == 0 or roi_h == 0:
                            print(f"警告: 数字 {digit_index} 清理后的ROI尺寸为零，跳过此数字。")
                            digit_index +=1
                            continue 
                        
                        if roi_w > roi_h:
                            new_w_roi = max_content_dim
                            new_h_roi = int(roi_h * (new_w_roi / roi_w))
                        else:
                            new_h_roi = max_content_dim
                            new_w_roi = int(roi_w * (new_h_roi / roi_h))
                        
                        new_w_roi = max(1, new_w_roi) # 确保尺寸至少为1
                        new_h_roi = max(1, new_h_roi) # 确保尺寸至少为1

                        # 使用 INTER_AREA 进行缩放，这可能会产生灰度图像
                        scaled_digit_roi_gray_cv = cv2.resize(cleaned_digit_roi_cv, (new_w_roi, new_h_roi), interpolation=cv2.INTER_AREA)
                        # 将缩放后的灰度图像重新二值化，以保持线条的清晰和二值特性
                        _, scaled_digit_roi_cv = cv2.threshold(scaled_digit_roi_gray_cv, 127, 255, cv2.THRESH_BINARY)
                        
                        cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"M7_scaled_roi_digit{digit_index}_INTER_AREA_thresh127_T{max_noise_area_threshold}.png"), scaled_digit_roi_cv)
                        
                        # M9: 准备模型输入 (同用户当前单数字逻辑的9)
                        paste_x = (IMAGE_SIZE - new_w_roi) // 2
                        paste_y = (IMAGE_SIZE - new_h_roi) // 2
                        final_mnist_like_image_np[paste_y : paste_y + new_h_roi, paste_x : paste_x + new_w_roi] = scaled_digit_roi_cv / 255.0
                        
                        Image.fromarray((final_mnist_like_image_np * 255).astype(np.uint8)).save(os.path.join(DEBUG_IMAGE_DIR, f"M9_final_for_model_digit{digit_index}_T{max_noise_area_threshold}.png"))

                        # 为当前数字进行预测
                        img_to_predict = final_mnist_like_image_np.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
                        prediction_vector = model.predict(img_to_predict)
                        predicted_digit = np.argmax(prediction_vector)
                        recognized_digits_list.append(str(predicted_digit))
                        print(f"识别到的数字 {digit_index} (轮廓基于5c) 预测为: {predicted_digit}")
                        
                        digit_index += 1

                    if recognized_digits_list:
                        self.result_label.config(text="".join(recognized_digits_list))
                        print(f"图片 '{filepath}' (OpenCV 多数字识别, 基于5c) 最终识别序列: {''.join(recognized_digits_list)}")
                    else: # candidate_contours非空但recognized_digits_list为空 (所有候选都无法处理)
                         self.result_label.config(text="?")
                         print("多数字识别：有候选轮廓，但未能成功处理和识别任何数字。")
                         # self.clear_processed_preview() # 保持显示M5e图

                    # 保存带有所有候选框的图像 (M5e)
                    if debug_candidates_display_img is not None:
                         cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, f"M5e_candidate_digits_on_5c_b{blockSize}_c{C_val}_T{max_noise_area_threshold}.png"), debug_candidates_display_img)
            
            else: # 如果在5c的输出中未找到任何初始轮廓 (all_potential_digit_contours is empty or None)
                print("多数字识别：在5c的输出中未找到任何初始轮廓。")
                self.result_label.config(text="?")
                self.clear_processed_preview()

            # 更新 "预处理后预览" 窗口，显示带有候选框的5c图像 (M5e, 如果生成了)
            if debug_candidates_display_img is not None:
                try:
                    processed_display_pil = Image.fromarray(cv2.cvtColor(debug_candidates_display_img, cv2.COLOR_BGR2RGB))
                    processed_display_pil.thumbnail((self.processed_image_display_size, self.processed_image_display_size), Image.Resampling.LANCZOS)
                    self.processed_tk_image_to_show = ImageTk.PhotoImage(processed_display_pil)
                    self.processed_image_label.config(image=self.processed_tk_image_to_show)
                    self.processed_image_label.image = self.processed_tk_image_to_show
                except Exception as e_display_multi:
                    print(f"在UI中显示多数字候选框图像时出错: {e_display_multi}")
                    self.clear_processed_preview()
            else: 
                self.clear_processed_preview()
            # --- 结束新的多数字识别逻辑 ---

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
