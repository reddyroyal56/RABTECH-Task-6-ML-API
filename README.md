
# RABTECH Academy Task 6
## Real-Time ML Inference REST API & Capstone

### Project Description
A Movie Review Sentiment Analysis REST API
built using Python, FastAPI and TensorFlow.

### Features
- Real-time sentiment prediction
- Positive and Negative classification
- Prediction confidence
- REST API using FastAPI
- Docker containerization
- Unit testing

### API Endpoints
- GET /
- GET /health
- POST /predict

### Run the Project

Install dependencies:

pip install -r requirements.txt

Start API:

uvicorn app:app --reload

Open documentation:

http://127.0.0.1:8000/docs

### Testing

pytest -v

### Docker

docker build -t sentiment-api .

docker run -p 8000:8000 sentiment-api

### Technology Used
Python, FastAPI, TensorFlow,
Scikit-learn, Docker and Pytest.
