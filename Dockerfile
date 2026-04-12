FROM python:3.11-slim
RUN pip install --no-cache-dir matplotlib torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app
CMD ["python", "draw.py"]
