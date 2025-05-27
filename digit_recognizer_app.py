import tkinter as tk
from tkinter import Canvas, Button, Label, Frame
from PIL import Image, ImageDraw, EpsImagePlugin
import numpy as np
import tensorflow as tf
import io

# 解决Pillow库的 Ghostscript 未安装的问题
# 有时在 Windows 上直接保存 Canvas 为 EPS 再用 Pillow 打开会依赖 Ghostscript
# 为了避免这个问题，我们先将 Canvas 内容绘制到 Pillow Image 对象上
EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs10.03.0\bin\gswin64c' # 设置你 Ghostscript 的实际路径

# --- 全局变量和常量 ---
CANVAS_WIDTH = 280
CANVAS_HEIGHT = 280
LINE_WIDTH = 15 # 调整画笔粗细，使其接近MNIST图像中的笔迹
IMAGE_SIZE = 28 # MNIST图像大小

# --- 模型加载 ---
try:
    model = tf.keras.models.load_model('model.h5')
    print("模型 model.h5 加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    model = None # 如果模型加载失败，将其设置为None

class DigitRecognizerApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("手写数字识别")
        self.root.resizable(False, False) # 禁止调整窗口大小

        # --- 画布区域 ---
        self.canvas_frame = Frame(self.root, bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas = Canvas(self.canvas_frame, bg="white", 
                             width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
                             cursor="crosshair") # 设置鼠标样式为十字线
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint) # 绑定鼠标左键拖动事件

        # 用于在内存中绘制图像，以便后续处理
        self.image = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), "white") # L表示灰度图
        self.draw = ImageDraw.Draw(self.image)

        # --- 控制和结果区域 ---
        self.controls_frame = Frame(self.root)
        self.controls_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        
        self.predict_button = Button(self.controls_frame, text="识别数字", 
                                     command=self.predict_digit, width=15, height=2)
        self.predict_button.pack(pady=10)

        self.clear_button = Button(self.controls_frame, text="清除画板", 
                                   command=self.clear_canvas, width=15, height=2)
        self.clear_button.pack(pady=10)

        self.result_label_title = Label(self.controls_frame, text="识别结果:", font=("Arial", 14))
        self.result_label_title.pack(pady=(20,5))
        
        self.result_label = Label(self.controls_frame, text="?", font=("Arial", 40, "bold"), 
                                  fg="blue", width=3, height=1, relief=tk.RIDGE)
        self.result_label.pack(pady=5)
        
        self.last_x, self.last_y = None, None # 用于存储上一个鼠标位置，以绘制连续线条

    def paint(self, event):
        """当鼠标在画板上拖动时绘制线条"""
        if self.last_x and self.last_y:
            # 在Tkinter Canvas上绘制（用户可见）
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=LINE_WIDTH, fill="black",
                                    capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
            # 同时在Pillow Image对象上绘制（用于后续处理）
            self.draw.line([self.last_x, self.last_y, event.x, event.y], 
                           fill="black", width=LINE_WIDTH, joint="round")
                           
        self.last_x = event.x
        self.last_y = event.y
    
    def canvas_reset_last_mouse_pos(self, event=None):
        """当鼠标按键释放或移出画布时，重置上一个鼠标位置，避免下次点击时画出错误的连接线"""
        self.last_x, self.last_y = None, None
        
    def clear_canvas(self):
        """清除画板和内存中的图像"""
        self.canvas.delete("all") # 清除Tkinter Canvas上的所有绘制内容
        # 重新创建 Pillow Image 和 Draw 对象，相当于清空
        self.image = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="?") # 重置结果显示
        print("画板已清除")

    def predict_digit(self):
        """对画板上的数字进行预处理并使用模型进行预测"""
        if model is None:
            self.result_label.config(text="错误")
            print("模型未加载，无法识别。")
            return

        try:
            # 1. 图像预处理
            #    Pillow图像已经是灰度图 ('L' mode)
            
            #    将画布图像 (CANVAS_WIDTH x CANVAS_HEIGHT) 缩小到 (IMAGE_SIZE x IMAGE_SIZE)
            #    使用ANTIALIASING (高质量缩放)
            img_resized = self.image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

            #    将Pillow图像转换为Numpy数组，像素值范围 [0, 255]
            img_array = np.array(img_resized)
            
            #    颜色反转: MNIST是黑底白字，我们的画布是白底黑字
            #    所以需要将像素值反转 (255 - pixel_value)
            img_array = 255 - img_array
            
            #    归一化: 将像素值从 [0, 255] 缩放到 [0, 1]
            img_array = img_array.astype('float32') / 255.0
            
            #    调整形状: 模型期望的输入形状是 (1, IMAGE_SIZE, IMAGE_SIZE, 1)
            #    1: 表示一个样本 (batch_size为1)
            #    IMAGE_SIZE, IMAGE_SIZE: 图像的宽和高
            #    1: 表示通道数 (灰度图为1)
            img_final = img_array.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

            # 2. 模型预测
            prediction = model.predict(img_final)
            predicted_digit = np.argmax(prediction) # 获取概率最高的类别索引

            # 3. 显示结果
            self.result_label.config(text=str(predicted_digit))
            print(f"预测结果: {predicted_digit}, 原始预测向量: {prediction}")

        except Exception as e:
            self.result_label.config(text="错误")
            print(f"识别过程中发生错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # 检查Ghostscript是否真的需要，如果不需要则移除相关代码或注释
    # 这个EpsImagePlugin的设置通常是为了解决tkinter canvas.postscript()的问题
    # 如果我们直接从内存中的Pillow Image对象处理，可能就不需要了
    # 但保留它以防某些环境下Pillow内部转换需要它
    try:
        img_test = Image.new("RGB", (10,10))
        img_test.save(io.BytesIO(), format='eps')
        print("EPS 保存测试成功 (可能不需要显式设置 Ghostscript 路径)")
    except Exception as e:
        print(f"EPS 保存测试失败，可能需要 Ghostscript: {e}")
        print(f"当前 Ghostscript 尝试路径: {EpsImagePlugin.gs_windows_binary}")
        print("如果程序因 Ghostscript 问题无法运行，请确保已安装 Ghostscript 并正确配置路径，")
        print("或者修改代码中 EpsImagePlugin.gs_windows_binary 的值为你的实际安装路径。")
        print("如果识别功能不依赖于EPS格式转换，可以尝试注释掉相关的EpsImagePlugin设置。")


    root = tk.Tk()
    app = DigitRecognizerApp(root)
    # 绑定鼠标释放和移出画布事件，用于重置last_x, last_y
    app.canvas.bind("<ButtonRelease-1>", app.canvas_reset_last_mouse_pos)
    app.canvas.bind("<Leave>", app.canvas_reset_last_mouse_pos) # 当鼠标移出画布时
    root.mainloop() 