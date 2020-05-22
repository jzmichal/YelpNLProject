FROM python:3.6
COPY app.py /deploy/
COPY templates/home.html /deploy/templates/
COPY static/css/yelp.css /deploy/static/css/
COPY requirements.txt /deploy/
COPY scaler.pkl /deploy/
COPY pca_transformer.pkl /deploy/
COPY rf_model.pkl /deploy/
WORKDIR /deploy/
RUN python3.6 -m pip install -r requirements.txt
RUN python3.6 -m pip install readability
EXPOSE 80
ENTRYPOINT ["python", "app.py"]
