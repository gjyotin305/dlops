FROM python:3.11-slim

WORKDIR /app/

COPY . .

RUN pip install uv 
RUN uv venv /app/dlops

ENV PATH="/app/dlops/bin:$PATH"

RUN uv pip install -r requirements.txt

CMD ["python", "train.py"]