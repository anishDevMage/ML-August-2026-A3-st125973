# A3 web app: Dash + the exported from-scratch logistic regression model
FROM python:3.10.12-slim-bookworm

WORKDIR /root/app

# install the dependencies first (Docker reuses this layer while requirements.txt does not change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# the app code and the exported model
COPY code/ ./code/
COPY model/ ./model/

ENV PORT=8050
EXPOSE 8050

CMD ["python3", "code/app.py"]
