CREATE DATABASE face_attendance
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE face_attendance;

CREATE TABLE admin (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL
);

CREATE TABLE attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    time DATETIME NOT NULL,
    date DATE NOT NULL
);

DROP TABLE IF EXISTS admin;

CREATE TABLE admin (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE,
    password VARCHAR(50)
);
INSERT INTO admin (username, password)
VALUES ('admin', '123456');
DELETE FROM admin WHERE username='admin';
