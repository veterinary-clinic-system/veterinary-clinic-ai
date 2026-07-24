# Not required for day-to-day development (prefer `uvicorn app.main:app --reload` on
# the host), but lets the whole stack run via `docker compose --profile full up` from
# veterinary-clinic-backend/.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
