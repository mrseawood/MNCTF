import os
import cv2
import torch
import shutil
import logging
import threading
import numpy as np
from PIL import Image
import tkinter as tk
from datetime import datetime
from tkinter import ttk, filedialog, messagebox
from transformers import CLIPProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

# 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化全局变量
input_folder = ""
output_folder = ""
processing_thread = None
use_gpu = True  # 添加GPU加速选项的默认值
device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

# 加载模型
logger.info("正在加载模型...")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to(device)
logger.info("模型加载完成")

class VideoProcessor:
    def __init__(self, video_path, output_folder, stop_event, threshold):
        self.video_path = video_path
        self.output_folder = output_folder
        self.stop_event = stop_event
        self.threshold = threshold
        self.nsfw_detected = False

    def process(self):
        try:
            logger.info(f"开始处理视频: {self.video_path}")
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {self.video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, round(fps))  # 每秒处理一帧
            frame_count = 0

            while cap.isOpened() and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    try:
                        if self.check_frame(frame):
                            self.nsfw_detected = True
                            logger.info(f"在视频中检测到敏感内容: {self.video_path}")
                            break
                    except Exception as e:
                        logger.error(f"处理帧时出错: {str(e)}")

                frame_count += 1

            cap.release()
            
            if self.nsfw_detected:
                self.move_video()

        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            raise RuntimeError(f"处理失败: {str(e)}")

    def check_frame(self, frame):
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image).resize((224, 224))
            
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = safety_checker(images=inputs.pixel_values, clip_input=inputs.pixel_values)
                # 详细记录safety_checker的输出结构
                logger.info(f"Safety Checker输出结构: {type(outputs)}, {len(outputs)}")
                logger.info(f"第一个输出: {outputs[0]}")
                logger.info(f"第二个输出: {outputs[1]}")
                
                # 获取NSFW检测结果并计算概率
                nsfw_values = outputs[1]
                if isinstance(nsfw_values, torch.Tensor):
                    nsfw_probs = float(nsfw_values.mean())  # 使用平均值作为概率
                else:
                    nsfw_probs = float(any(nsfw_values))  # 如果不是tensor，则使用any
                
                logger.info(f"NSFW概率值: {nsfw_probs}, 阈值: {self.threshold}")
                return nsfw_probs > self.threshold
        except Exception as e:
            logger.error(f"检查帧时出错: {str(e)}")
            return False

    def move_video(self):
        try:
            filename = os.path.basename(self.video_path)
            dest_path = os.path.join(self.output_folder, filename)
            
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while True:
                    new_name = f"{base}_({counter}){ext}"
                    new_path = os.path.join(self.output_folder, new_name)
                    if not os.path.exists(new_path):
                        dest_path = new_path
                        break
                    counter += 1
            
            shutil.move(self.video_path, dest_path)
            logger.info(f"已移动视频到: {dest_path}")
        except Exception as e:
            logger.error(f"移动视频时出错: {str(e)}")
            raise

class ProcessingThread(threading.Thread):
    def __init__(self, window, threshold):
        super().__init__()
        self.window = window
        self.stop_event = threading.Event()
        self.threshold = threshold
        self.current_file = ""
        self.progress = 0
        self.processed_count = 0
        self.total_count = 0

    def run(self):
        try:
            self.process_directory(input_folder)

            if not self.stop_event.is_set():
                logger.info(f"处理完成，成功处理 {self.processed_count}/{self.total_count} 个文件")
                self.window.event_generate("<<Complete>>", when="tail")

        except Exception as e:
            logger.error(f"处理线程出错: {str(e)}")
            self.window.event_generate("<<Error>>", when="tail", data=str(e))

    def process_directory(self, current_dir):
        try:
            # 获取所有视频文件和子文件夹
            all_items = os.listdir(current_dir)
            video_files = [f for f in all_items if os.path.isfile(os.path.join(current_dir, f)) and 
                          f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".flv"))]
            subdirs = [d for d in all_items if os.path.isdir(os.path.join(current_dir, d))]

            # 更新总文件计数
            self.total_count += len(video_files)

            # 处理当前目录中的视频文件
            for filename in video_files:
                if self.stop_event.is_set():
                    logger.info("处理被用户取消")
                    return

                self.current_file = filename
                self.progress = (self.processed_count / max(1, self.total_count)) * 100
                self.window.event_generate("<<ProgressUpdate>>", when="tail")

                video_path = os.path.join(current_dir, filename)
                # 计算相对路径，用于在输出目录中创建相同的结构
                rel_path = os.path.relpath(current_dir, input_folder)
                target_dir = os.path.join(output_folder, rel_path) if rel_path != '.' else output_folder

                try:
                    # 确保目标目录存在
                    os.makedirs(target_dir, exist_ok=True)
                    processor = VideoProcessor(video_path, target_dir, self.stop_event, self.threshold)
                    processor.process()
                    self.processed_count += 1
                except Exception as e:
                    logger.error(f"处理视频 {filename} 时出错: {str(e)}")

            # 递归处理子文件夹
            for subdir in subdirs:
                if self.stop_event.is_set():
                    return
                subdir_path = os.path.join(current_dir, subdir)
                self.process_directory(subdir_path)

        except Exception as e:
            logger.error(f"处理目录 {current_dir} 时出错: {str(e)}")
            raise
class VideoNSFWDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("视频安全检测系统")
        self.setup_ui()
        self.setup_event_handlers()

    def setup_ui(self):
        # 设置窗口样式
        style = ttk.Style()
        style.configure("TButton", padding=6)
        style.configure("TFrame", padding=10)

        # 控制面板
        control_frame = ttk.Frame(self.root, padding="10 10 10 10")
        control_frame.pack(fill=tk.X)

        # 文件夹选择
        folder_frame = ttk.Frame(control_frame)
        folder_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(folder_frame, text="选择输入文件夹", command=self.select_input_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(folder_frame, text="选择输出文件夹", command=self.select_output_folder).pack(side=tk.LEFT, padx=5)
        
        self.input_label = ttk.Label(control_frame, text="输入文件夹: 未选择")
        self.input_label.pack(fill=tk.X, pady=2)
        
        self.output_label = ttk.Label(control_frame, text="输出文件夹: 未选择")
        self.output_label.pack(fill=tk.X, pady=2)

        # 阈值设置
        threshold_frame = ttk.Frame(control_frame)
        threshold_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(threshold_frame, text="敏感度阈值 (0-1):").pack(side=tk.LEFT)
        self.threshold_slider = ttk.Scale(threshold_frame, from_=0, to=1, orient=tk.HORIZONTAL)
        self.threshold_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.threshold_slider.set(0.7)
        
        self.threshold_entry = ttk.Entry(threshold_frame, width=5)
        self.threshold_entry.pack(side=tk.LEFT)
        self.threshold_entry.insert(0, "0.7")

        # 添加阈值说明
        threshold_desc = ttk.Label(control_frame, text="阈值说明：\n较高的阈值（>0.7）：检测更严格，仅移动高度可疑的视频\n中等阈值（0.4-0.7）：平衡的检测灵敏度\n较低的阈值（<0.4）：检测更宽松，可能会移动更多的视频", justify=tk.LEFT)
        threshold_desc.pack(fill=tk.X, pady=(5, 10))

        # GPU加速选项
        gpu_frame = ttk.Frame(control_frame)
        gpu_frame.pack(fill=tk.X, pady=5)
        
        self.use_gpu_var = tk.BooleanVar(value=True)
        self.gpu_checkbox = ttk.Checkbutton(gpu_frame, text="启用GPU加速（如果可用）", variable=self.use_gpu_var, command=self.toggle_gpu)
        self.gpu_checkbox.pack(side=tk.LEFT)
        
        if not torch.cuda.is_available():
            self.use_gpu_var.set(False)
            self.gpu_checkbox.config(state=tk.DISABLED)
            ttk.Label(gpu_frame, text="（未检测到可用的GPU）", foreground="gray").pack(side=tk.LEFT, padx=5)

        # 进度条和状态信息
        progress_frame = ttk.Frame(self.root)
        progress_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress_bar.pack(fill=tk.X)
        
        self.status_label = ttk.Label(progress_frame, text="准备就绪")
        self.status_label.pack(pady=5)

        # 操作按钮
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="开始处理", command=self.start_processing)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.cancel_btn = ttk.Button(btn_frame, text="取消处理", state=tk.DISABLED, command=self.cancel_processing)
        self.cancel_btn.pack(side=tk.LEFT, padx=5)

        # 绑定事件
        self.threshold_entry.bind("<Return>", self.update_slider)
        self.threshold_slider.bind("<Motion>", self.update_entry)

    def setup_event_handlers(self):
        self.root.bind("<<ProgressUpdate>>", self.update_progress)
        self.root.bind("<<Complete>>", self.on_complete)
        self.root.bind("<<Error>>", self.on_error)

    def update_slider(self, event):
        try:
            value = float(self.threshold_entry.get())
            if 0 <= value <= 1:
                self.threshold_slider.set(value)
        except ValueError:
            pass

    def update_entry(self, event):
        value = self.threshold_slider.get()
        self.threshold_entry.delete(0, tk.END)
        self.threshold_entry.insert(0, f"{value:.2f}")

    def select_input_folder(self):
        global input_folder
        folder = filedialog.askdirectory()
        if folder:
            input_folder = folder
            self.input_label.config(text=f"输入文件夹: {input_folder}")
            logger.info(f"已选择输入文件夹: {input_folder}")

    def select_output_folder(self):
        global output_folder
        folder = filedialog.askdirectory()
        if folder:
            output_folder = folder
            self.output_label.config(text=f"输出文件夹: {output_folder}")
            logger.info(f"已选择输出文件夹: {output_folder}")

    def start_processing(self):
        global processing_thread
        if not input_folder or not output_folder:
            messagebox.showerror("错误", "请先选择输入和输出文件夹")
            return
        
        try:
            threshold = float(self.threshold_entry.get())
            if not (0 <= threshold <= 1):
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "阈值必须是0到1之间的数字")
            return

        self.start_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.status_label.config(text="正在处理...")
        
        processing_thread = ProcessingThread(self.root, threshold)
        processing_thread.start()
        logger.info("开始处理视频")

    def cancel_processing(self):
        if processing_thread:
            processing_thread.stop_event.set()
            processing_thread.join()
            logger.info("处理已取消")
        self.reset_ui()
        messagebox.showinfo("信息", "处理已取消")

    def update_progress(self, event):
        self.progress_bar["value"] = processing_thread.progress
        self.status_label.config(text=f"正在处理: {processing_thread.current_file}")
        self.root.update_idletasks()

    def on_complete(self, event):
        self.reset_ui()
        messagebox.showinfo("完成", "视频处理完成！")
        self.progress_bar["value"] = 100
        logger.info("所有视频处理完成")

    def on_error(self, event):
        error_msg = event.data
        self.reset_ui()
        messagebox.showerror("处理错误", f"发生错误:\n{error_msg}")
        logger.error(f"处理错误: {error_msg}")

    def reset_ui(self):
        self.start_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.status_label.config(text="准备就绪")
        self.progress_bar["value"] = 0

    def toggle_gpu(self):
        global device, safety_checker
        use_gpu = self.use_gpu_var.get()
        device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        safety_checker = safety_checker.to(device)
        logger.info(f"切换到{device}设备")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = VideoNSFWDetector(root)
        logger.info("应用程序启动")
        root.mainloop()
    except Exception as e:
        logger.critical(f"应用程序发生严重错误: {str(e)}")
        raise