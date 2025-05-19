Here's your content in **Markdown format**:

# CXR Inference Backend

A FastAPI backend for chest X-ray (CXR) analysis using a PyTorch DenseNet121 model and Gemini API for report generation.

---

## 🚀 Setup

### 1. Clone the repository:

```bash
git clone https://github.com/sameeramjad07/cxr-backend.git
cd cxr-backend
```

### 2. Initiliaze and activate your vitual environment:

```bash
python -m venv <your-env-name>
.\<your-env-name>\Scripts\activate
```



### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Set up environment:

- Copy `.env.example` to `.env`
- Update `GEMINI_API_KEY` with your Gemini API key.

### 5. Place model:

- Copy `model_epoch_16.pth` to `app/models/`.

### 6. Run the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 🧪 Test Endpoints

- **Health check**:

  ```bash
  curl http://localhost:8000/health
  ```

- **Predict**:

  ```bash
  curl -X POST -F "file=@image.jpg" http://localhost:8000/predict
  ```

- **Run tests**:

  ```bash
  python -m unittest discover tests
  ```

---

## 🔌 Endpoints

- `POST /predict`: Upload a CXR image and get predictions (probabilities for 14 conditions) and inference time.
- `POST /generate-report`: Generate a PDF report based on predictions.
- `GET /health`: Check server status.

---

## 🔗 Integration with Next.js

Update your Next.js frontend to call:

- `http://localhost:8000/predict`
- `http://localhost:8000/generate-report`

