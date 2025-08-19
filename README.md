# 🌸 Iris Flower Classification — FastAPI ML Inference ## 📝 Problem Description The goal of this project is to classify iris flowers into one of three species: - Setosa - Versicolor - Virginica based on four flower measurements: - Sepal length (cm) - Sepal width (cm) - Petal length (cm) - Petal width (cm) --- ## 🤖 Model Choice Justification We use **Logistic Regression** inside a scikit-learn **Pipeline** with StandardScaler: - Works very well for small, structured datasets like Iris. - Simple, fast, and interpretable model. - Achieves ~96–97% accuracy on test data. - Provides probability estimates (predict_proba) so we can report prediction confidence. --- ## 📡 API Usage Examples ### 1. Health check GET / Example response:
json
{
  "status": "healthy",
  "message": "Iris API running"
}
--- ### 2. Predict a single flower POST /predict
json

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
Response:
json

{
  "prediction": "setosa",
  "confidence": 0.99
}
### 3. Model info GET /model-info Example response:
json
{
  "model_type": "LogisticRegression (with StandardScaler)",
  "features": ["sepal_length","sepal_width","petal_length","petal_width"],
  "class_names": ["setosa","versicolor","virginica"],
  "metrics": {"accuracy": 0.9667}
}
### 4. Bonus: Batch prediction POST /predict-batch Request:
json

{
  "items": [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.2, "sepal_width": 3.4, "petal_length": 5.4, "petal_width": 2.3}
  ]
}
Response:
json

{
  "predictions": [
    {"prediction": "setosa", "confidence": 0.99},
    {"prediction": "virginica", "confidence": 0.97}
  ]
}
### 🚀 How to Run the Application Setup environment
bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
Generate dataset
bash python make_dataset.py This creates data/iris.csv. Train the model
bash

python train_model.py
This creates:

model.pkl (trained model)

model_meta.json (metadata)

Start FastAPI server
bash uvicorn main:app --reload Server will be available at: 👉 http://127.0.0.1:8000 Interactive API docs: 👉 http://127.0.0.1:8000/docs Test page (bonus): 👉 http://127.0.0.1:8000/test 📈 Results Model achieves ~96–97% accuracy on test data. Predictions include confidence score from predict_proba. ✅ Deliverables main.py → FastAPI app (endpoints) train_model.py → training script make_dataset.py → dataset generator requirements.txt → dependencies README.md → documentation model.pkl + model_meta.json → trained model & metadata Dataset: data/iris.csv ### 📊 Project Workflow
pgsql

Dataset (iris.csv) → Training (train_model.py) → Model (model.pkl + meta.json)
       ↓
   FastAPI (main.py)
       ↓
 REST Endpoints (predict, model-info, etc.)
       ↓
     User / Client correct this markdown