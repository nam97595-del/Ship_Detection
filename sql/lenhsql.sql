-- Tạo Database shipdb
CREATE DATABASE shipdb;
GO
USE shipdb;
GO

-- Tạo bảng users
CREATE TABLE users (
    id INT IDENTITY(1,1) PRIMARY KEY,
    username NVARCHAR(50) NOT NULL UNIQUE,
    password NVARCHAR(255) NOT NULL,
    full_name NVARCHAR(100)
);
GO

-- Thêm tài khoản mẫu
INSERT INTO users (username, password, full_name)
VALUES ('admin', '123', 'Quản trị viên');
GO


CREATE TABLE shiplog (
    id INT IDENTITY(1,1) PRIMARY KEY,
    track_id INT NOT NULL,           
    class_name NVARCHAR(100),         
    so_hieu NVARCHAR(50) DEFAULT NULL, 
    gio_phat_hien DATETIME DEFAULT GETDATE(),
    hinh_anh_path NVARCHAR(255)       
);
GO

select * from shiplog