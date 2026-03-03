import pyodbc
import os
from typing import Optional


class DatabaseConnection:
    """
    Class quản lý kết nối đến SQL Server (có thể dùng singleton hoặc context manager)
    """

    _instance = None
    _conn: Optional[pyodbc.Connection] = None

    def __new__(cls):
        """Singleton pattern - chỉ tạo một instance duy nhất"""
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Chỉ khởi tạo lần đầu
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.server = r'.\SQLEXPRESS'          # có thể thay bằng biến môi trường
            self.database = 'shipdb'
            self.driver = 'ODBC Driver 17 for SQL Server'

    def get_connection_string(self) -> str:
        """Trả về chuỗi kết nối chuẩn"""
        return (
            f'DRIVER={{{self.driver}}};'
            f'SERVER={self.server};'
            f'DATABASE={self.database};'
            'Trusted_Connection=yes;'
            # Nếu dùng tài khoản SQL: thêm UID=...;PWD=...
        )

    def connect(self) -> Optional[pyodbc.Connection]:
        """Tạo hoặc trả về kết nối hiện có"""
        if self._conn is None or self._conn.closed:
            try:
                self._conn = pyodbc.connect(self.get_connection_string())
                print(">> Kết nối database thành công")
            except pyodbc.Error as e:
                print(f">> Lỗi kết nối database: {e}")
                self._conn = None
        return self._conn

    def close(self):
        """Đóng kết nối nếu đang mở"""
        if self._conn and not self._conn.closed:
            self._conn.close()
            print(">> Đã đóng kết nối database")
            self._conn = None

    def __enter__(self):
        """Hỗ trợ dùng với with statement"""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Tự động commit hoặc rollback và đóng kết nối"""
        if self._conn:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
            self.close()


# Global instance tiện dùng (singleton)
db = DatabaseConnection()


# Hàm tiện ích nhanh
def get_db_connection() -> Optional[pyodbc.Connection]:
    """Hàm tiện ích để lấy connection nhanh"""
    return db.connect()


def close_db_connection():
    """Đóng kết nối toàn cục nếu cần"""
    db.close()


# Ví dụ sử dụng với context manager (khuyến nghị)
def with_connection():
    with DatabaseConnection() as conn:
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT @@VERSION")
            print(cursor.fetchone())