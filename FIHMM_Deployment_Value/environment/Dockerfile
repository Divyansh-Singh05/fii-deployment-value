# Reproduction image. Fully self-contained: the dataset ships inside the build
# context under data/, so nothing is mounted, fetched or configured at run time.
#
#   docker build -f environment/Dockerfile -t fihmm .
#   docker run --rm fihmm                                  # verify
#   docker run --rm fihmm python reproduction/reproduce_all.py
#
# The raw depository record is NOT included and cannot be: it is proprietary.
# It is not needed - data/derived carries the panels the reproduction consumes.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

WORKDIR /app
COPY environment/requirements.txt /app/environment/requirements.txt
RUN pip install --no-cache-dir -r environment/requirements.txt
COPY . /app

CMD ["python", "reproduction/verify_results.py"]
