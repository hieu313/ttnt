# Sử dụng Python 3.9 làm base image
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy requirements.txt và cài đặt dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir result
# Copy toàn bộ code vào container
COPY . .

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1

# Command mặc định khi chạy container
CMD ["python", "main.py"]
