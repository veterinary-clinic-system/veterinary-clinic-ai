FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# opencv-python can thu vien he thong libGL/glib de import cv2
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ma tran bi ghi lai khi chay (online learning, sua tren dashboard) nen nam tren
# volume /app/state; MatrixEngine tu chep ban CSV di kem image vao khi volume trong.
ENV MATRIX_CSV_PATH=/app/state/symptom_disease_matrix.csv
VOLUME /app/state

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
