# -*- coding: utf-8 -*-
"""
比较两个PNG水印图片的NC值
通过弹窗选择文件并计算标准化相关系数(NC)
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 导入项目中的NC计算函数
from NC import NC, image_to_array

class WatermarkComparator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("水印图片NC值比较工具")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # 设置中文字体
        try:
            import matplotlib
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
        except:
            pass
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主标题
        title_label = tk.Label(self.root, text="水印图片NC值比较工具", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # 说明文字
        info_label = tk.Label(self.root, 
                             text="请选择两个PNG格式的水印图片进行NC值比较\n"
                                  "NC值范围：0-1，值越接近1表示相似度越高", 
                             font=("Arial", 10), 
                             justify=tk.CENTER)
        info_label.pack(pady=10)
        
        # 第一个图片选择框
        frame1 = tk.Frame(self.root)
        frame1.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(frame1, text="原始水印图片:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.path1_var = tk.StringVar()
        self.path1_entry = tk.Entry(frame1, textvariable=self.path1_var, 
                                   font=("Arial", 10), state="readonly")
        self.path1_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        btn1 = tk.Button(frame1, text="选择文件", command=self.select_image1,
                        bg="#4CAF50", fg="white", font=("Arial", 10))
        btn1.pack(side=tk.RIGHT)
        
        # 第二个图片选择框
        frame2 = tk.Frame(self.root)
        frame2.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(frame2, text="提取水印图片:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.path2_var = tk.StringVar()
        self.path2_entry = tk.Entry(frame2, textvariable=self.path2_var, 
                                   font=("Arial", 10), state="readonly")
        self.path2_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        btn2 = tk.Button(frame2, text="选择文件", command=self.select_image2,
                        bg="#4CAF50", fg="white", font=("Arial", 10))
        btn2.pack(side=tk.RIGHT)
        
        # 计算按钮
        calc_btn = tk.Button(self.root, text="计算NC值", command=self.calculate_nc,
                           bg="#2196F3", fg="white", font=("Arial", 12, "bold"),
                           pady=10)
        calc_btn.pack(pady=20)
        
        # 结果显示区域
        result_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=2)
        result_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        tk.Label(result_frame, text="计算结果:", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=10, pady=5)
        
        self.result_text = tk.Text(result_frame, height=8, font=("Courier", 11), 
                                  state=tk.DISABLED, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # 底部按钮框
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(pady=10)
        
        # 显示图片按钮
        show_btn = tk.Button(bottom_frame, text="显示图片对比", command=self.show_images,
                           bg="#FF9800", fg="white", font=("Arial", 10))
        show_btn.pack(side=tk.LEFT, padx=5)
        
        # 清除按钮
        clear_btn = tk.Button(bottom_frame, text="清除", command=self.clear_all,
                            bg="#f44336", fg="white", font=("Arial", 10))
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # 退出按钮
        exit_btn = tk.Button(bottom_frame, text="退出", command=self.root.quit,
                           bg="#9E9E9E", fg="white", font=("Arial", 10))
        exit_btn.pack(side=tk.LEFT, padx=5)
        
    def select_image1(self):
        """选择第一个图片"""
        filename = filedialog.askopenfilename(
            title="选择原始水印图片",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        if filename:
            self.path1_var.set(filename)
            self.log_message(f"已选择原始水印图片: {os.path.basename(filename)}")
    
    def select_image2(self):
        """选择第二个图片"""
        filename = filedialog.askopenfilename(
            title="选择提取水印图片",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        if filename:
            self.path2_var.set(filename)
            self.log_message(f"已选择提取水印图片: {os.path.basename(filename)}")
    
    def log_message(self, message):
        """在结果区域显示消息"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.config(state=tk.DISABLED)
        self.result_text.see(tk.END)
        self.root.update()
    
    def calculate_nc(self):
        """计算NC值"""
        path1 = self.path1_var.get()
        path2 = self.path2_var.get()
        
        if not path1 or not path2:
            messagebox.showerror("错误", "请先选择两个PNG图片文件！")
            return
        
        if not os.path.exists(path1):
            messagebox.showerror("错误", f"文件不存在: {path1}")
            return
            
        if not os.path.exists(path2):
            messagebox.showerror("错误", f"文件不存在: {path2}")
            return
        
        try:
            self.log_message("=" * 50)
            self.log_message("开始计算NC值...")
            
            # 读取图片并转换为数组
            self.log_message("正在读取原始水印图片...")
            img1_array = image_to_array(path1)
            self.log_message(f"原始水印尺寸: {img1_array.shape}")
            
            self.log_message("正在读取提取水印图片...")
            img2_array = image_to_array(path2)
            self.log_message(f"提取水印尺寸: {img2_array.shape}")
            
            # 检查图片尺寸是否一致
            if img1_array.shape != img2_array.shape:
                self.log_message(f"警告: 图片尺寸不一致！")
                self.log_message(f"原始水印: {img1_array.shape}")
                self.log_message(f"提取水印: {img2_array.shape}")
                messagebox.showwarning("警告", "两个图片的尺寸不一致，无法计算NC值！")
                return
            
            # 计算NC值
            self.log_message("正在计算NC值...")
            nc_value = NC(img1_array, img2_array)
            
            # 显示结果
            self.log_message("=" * 50)
            self.log_message("计算完成！")
            self.log_message("=" * 50)
            self.log_message(f"NC值: {nc_value:.6f}")
            
            # 根据NC值给出评估
            if nc_value >= 0.95:
                quality = "优秀"
                color = "green"
            elif nc_value >= 0.9:
                quality = "良好"
                color = "blue"
            elif nc_value >= 0.8:
                quality = "一般"
                color = "orange"
            elif nc_value >= 0.7:
                quality = "较差"
                color = "red"
            else:
                quality = "很差"
                color = "darkred"
            
            self.log_message(f"相似度等级: {quality}")
            self.log_message("=" * 50)
            
            # 显示详细信息
            self.log_message("详细信息:")
            self.log_message(f"原始水印文件: {os.path.basename(path1)}")
            self.log_message(f"提取水印文件: {os.path.basename(path2)}")
            self.log_message(f"图片尺寸: {img1_array.shape}")
            self.log_message(f"图片类型: PNG")
            
            # 计算其他统计信息
            total_pixels = img1_array.size
            same_pixels = np.sum(img1_array == img2_array)
            diff_pixels = total_pixels - same_pixels
            accuracy = same_pixels / total_pixels * 100
            
            self.log_message(f"总像素数: {total_pixels}")
            self.log_message(f"相同像素数: {same_pixels}")
            self.log_message(f"不同像素数: {diff_pixels}")
            self.log_message(f"像素匹配率: {accuracy:.2f}%")
            
            # 显示成功消息框
            messagebox.showinfo("计算完成", 
                              f"NC值计算完成！\n\n"
                              f"NC值: {nc_value:.6f}\n"
                              f"相似度等级: {quality}\n"
                              f"像素匹配率: {accuracy:.2f}%")
            
        except Exception as e:
            error_msg = f"计算过程中发生错误: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("错误", error_msg)
    
    def show_images(self):
        """显示图片对比"""
        path1 = self.path1_var.get()
        path2 = self.path2_var.get()
        
        if not path1 or not path2:
            messagebox.showerror("错误", "请先选择两个PNG图片文件！")
            return
        
        try:
            # 读取图片
            img1 = Image.open(path1)
            img2 = Image.open(path2)
            
            # 创建对比图
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 显示原始水印
            axes[0].imshow(img1, cmap='gray')
            axes[0].set_title(f'原始水印\n{os.path.basename(path1)}', fontsize=12)
            axes[0].axis('off')
            
            # 显示提取水印
            axes[1].imshow(img2, cmap='gray')
            axes[1].set_title(f'提取水印\n{os.path.basename(path2)}', fontsize=12)
            axes[1].axis('off')
            
            # 显示差异图
            if img1.size == img2.size:
                img1_array = np.array(img1)
                img2_array = np.array(img2)
                diff = np.abs(img1_array.astype(int) - img2_array.astype(int))
                axes[2].imshow(diff, cmap='hot')
                axes[2].set_title('差异图\n(红色=不同, 黑色=相同)', fontsize=12)
            else:
                axes[2].text(0.5, 0.5, '尺寸不一致\n无法生成差异图', 
                           ha='center', va='center', transform=axes[2].transAxes)
                axes[2].set_title('差异图', fontsize=12)
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            error_msg = f"显示图片时发生错误: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("错误", error_msg)
    
    def clear_all(self):
        """清除所有内容"""
        self.path1_var.set("")
        self.path2_var.set("")
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.log_message("已清除所有内容，请重新选择图片。")
    
    def run(self):
        """运行应用"""
        # 添加一些使用说明
        self.log_message("欢迎使用水印图片NC值比较工具！")
        self.log_message("使用说明:")
        self.log_message("1. 点击'选择文件'按钮选择两个PNG格式的水印图片")
        self.log_message("2. 点击'计算NC值'开始计算")
        self.log_message("3. 可以点击'显示图片对比'查看图片差异")
        self.log_message("4. NC值范围0-1，越接近1表示相似度越高")
        self.log_message("=" * 50)
        
        self.root.mainloop()

def main():
    """主函数"""
    try:
        app = WatermarkComparator()
        app.run()
    except Exception as e:
        print(f"程序启动失败: {str(e)}")
        messagebox.showerror("错误", f"程序启动失败: {str(e)}")

if __name__ == "__main__":
    main()
