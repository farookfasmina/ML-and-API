🌸 Iris Flower Classification — FastAPI ML Inference

 📝 Problem Description
The goal of this project is to classify iris flowers into one of three species:
- Setosa
- Versicolor
- Virginica

based on four measurements:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

---

 🤖 Model Choice Justification
We use **Logistic Regression** within a **scikit-learn Pipeline** with `StandardScaler`:
- Simple and effective for small, linearly separable datasets like Iris.
- Fast to train and easy to interpret.
- Achieves ~96–97% accuracy on test data, which is sufficient for this problem.
- Supports probability estimates (`predict_proba`) to give confidence scores.

---

📡 API Usage Examples

 1. Health check
`GET /`  
Response:
```json
{"status":"healthy","message":"Iris API running"}
2. Predict one flower
POST /predict
Request:

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
Response:

{
  "prediction": "setosa",
  "confidence": 0.99
}
3. Model info
GET /model-info
Example response:

{
  "model_type": "LogisticRegression (with StandardScaler)",
  "features": ["sepal_length","sepal_width","petal_length","petal_width"],
  "class_names": ["setosa","versicolor","virginica"],
  "metrics": {"accuracy": 0.9667}
}
🚀 How to Run the Application
1. Setup environment

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
2. Prepare dataset
python make_dataset.py

3. Train model
python train_model.py

This creates:

model.pkl (trained model)

model_meta.json (metadata)

4. Run FastAPI server
uvicorn main:app --reload
Server will be available at:
👉 http://127.0.0.1:8000

Interactive API docs:
👉 http://127.0.0.1:8000/docs

📈 Results
Model achieves ~96–97% accuracy on test split.

Confidence score provided from predict_proba.

✅ Deliverables
main.py → FastAPI app

train_model.py → training script

requirements.txt → dependencies

README.md → documentation

model.pkl + model_meta.json → trained model + metadata

Dataset in data/iris.csv