FROM python:3.10-slim

WORKDIR /app

# Copy requirements from docker folder, rename to root
COPY docker/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src

# Copy app.py from docker folder, rename to root
COPY docker/app.py ./app.py

# Hugging Face Spaces require port 7860
EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]

