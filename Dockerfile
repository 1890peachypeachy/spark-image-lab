FROM nvcr.io/nvidia/pytorch:26.08-py3
COPY requirements.txt security-requirements.txt /opt/spark-image-lab/
RUN python -m pip install --no-cache-dir --ignore-installed --no-deps \
        -r /opt/spark-image-lab/security-requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/spark-image-lab/requirements.txt
ENV HF_HOME=/workspace/cache/huggingface PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1 SPARK_HOST=0.0.0.0
WORKDIR /workspace
CMD ["python", "lab.py", "--serve"]
