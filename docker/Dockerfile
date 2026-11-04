FROM python:3.11-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/main.py /app/app/main.py

COPY models/lcnet100_bifpn_4d.ncnn.param /app/models/
COPY models/lcnet100_bifpn_4d.ncnn.bin /app/models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--reload-dir", "/app/app"]