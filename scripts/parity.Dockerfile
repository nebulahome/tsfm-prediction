# The official Python implementation, CPU-only, for scripts/parity.py.
# docker build -f scripts/parity.Dockerfile -t tsfm-parity scripts
FROM python:3.12-slim
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir "timestral @ https://github.com/mistralai/timestral/archive/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff.tar.gz"
COPY parity.py /parity.py
ENTRYPOINT ["python", "/parity.py"]
