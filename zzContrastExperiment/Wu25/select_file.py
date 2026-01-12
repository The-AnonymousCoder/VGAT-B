# -*- coding: utf-8 -*-
# @Time    : 2023/11/3 19:35
# @Author  :Fivem
# @File    : select_file.py
# @Software: PyCharm
# @last modified:2023/11/3 19:35
import tkinter as tk
from tkinter import filedialog


def select_file(title, file_type):
    """
    filter file
    :param title: ex:'select original watermark',
    :param file_type: ex:[("watermark file", "*.png *.jpg")]
    :return: file path
    """
    # 创建主窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    # 打开文件选择对话框，并应用筛选条件
    file_path = filedialog.askopenfilename(title=title, filetypes=file_type)
    # 关闭主窗口
    root.destroy()
    # 检查用户是否选择了文件
    if file_path:
        return file_path.replace('/', '\\')
    else:
        exit("no select any file")


def select_folder():
    """
    Select a folder using file dialog
    :param :
    :return: Selected folder path
    """
    # 创建主窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    # 打开文件夹选择对话框
    folder_path = filedialog.askdirectory()
    # 关闭主窗口
    root.destroy()
    # 检查用户是否选择了文件夹
    if folder_path:
        return folder_path.replace('/', '\\')
    else:
        exit("no select any folder")


if __name__ == "__main__":
    folder_path = select_folder()
