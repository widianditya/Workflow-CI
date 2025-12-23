# Gunakan base image Python yang ringan
FROM python:3.12-slim

# Set folder kerja di dalam container
WORKDIR /app

# Salin file requirements dan instal library
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh isi folder MLProject ke dalam container
COPY . .

# Perintah untuk menjalankan aplikasi (sesuaikan jika Anda menggunakan flask/fastapi)
# Jika hanya untuk menyimpan model, biarkan seperti ini
CMD ["python", "modelling.py"]