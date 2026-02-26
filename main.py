import tkinter as tk
import sys
import os

# Thêm đường dẫn gốc để tránh lỗi ModuleNotFoundError khi import từ các thư mục con
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.login import LoginWindow
from gui.main_window import App

def start_main_app():
    """Hàm này sẽ chạy sau khi người dùng đăng nhập thành công"""
    # Khởi tạo cửa sổ chính của ứng dụng phát hiện tàu
    main_root = tk.Tk()
    app = App(main_root)
    
    # Đảm bảo tắt tiến trình ngầm khi đóng cửa sổ
    main_root.protocol("WM_DELETE_WINDOW", app.on_closing)
    main_root.mainloop()

if __name__ == "__main__":
    # Bước 1: Khởi tạo cửa sổ Đăng nhập trước
    login_root = tk.Tk()
    
    # Bước 2: Truyền hàm start_main_app vào làm success_callback
    # Khi LoginWindow gọi self.success_callback(), nó sẽ thực thi hàm start_main_app
    login_app = LoginWindow(login_root, success_callback=start_main_app)
    
    login_root.mainloop()