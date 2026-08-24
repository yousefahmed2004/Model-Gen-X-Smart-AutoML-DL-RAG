
# 🤖 Model Gen X — Smart AutoML, DL & RAG Agent

> **Build, train, evaluate, and interact with Machine Learning, Deep Learning, and RAG models using AI — without the complexity of traditional ML workflows.**

Model Gen X is an AI-powered machine learning platform that allows users to interact with an intelligent agent, upload datasets, describe what they want to build, train models, evaluate results, test trained models, and build RAG-powered applications.

The platform combines **AutoML, Deep Learning, RAG, AI Agents, and an interactive web interface** into a unified AI development environment.

---

## 🚀 What is Model Gen X?

Traditional machine learning workflows require users to manually:

- Prepare datasets
- Clean and preprocess data
- Select algorithms
- Configure hyperparameters
- Train models
- Evaluate results
- Save models
- Build prediction interfaces
- Create RAG pipelines

**Model Gen X simplifies this process.**

Users can interact with an AI assistant, upload their data, explain what they want to predict, and let the platform handle the machine learning workflow.

```text
                    USER
                      │
                      ▼
                AI Assistant
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       AutoML       Deep       RAG
       Training    Learning   Pipeline
          │           │           │
          └───────────┼───────────┘
                      ▼
                Trained Model
                      │
                      ▼
                 Playground
                      │
                      ▼
                 Predictions
````

---

# ✨ Key Features

* 🤖 AI-powered ML assistant
* 🧠 AI Agent for model training
* ⚡ Automated Machine Learning
* 🧬 Deep Learning workflows
* 📊 Dataset upload and analysis
* 🔍 Classification and Regression
* 🏆 Automated model selection
* 📈 Model evaluation
* 🧪 Machine Learning model playground
* 🧠 Deep Learning model playground
* 📚 RAG pipeline
* 💬 RAG-powered chatbot
* 📁 Training document upload
* 🔎 Retrieval-Augmented Generation
* 💾 Model persistence
* 📥 Model download
* 🔐 Authentication
* 🌐 Interactive web platform
* 🌙 Dark / Light mode
* 🌍 English / Arabic support
* 📱 Responsive interface

---

# 🧠 AI Assistant

The platform includes an AI assistant that helps users interact with the ML system using natural language.

![AI Assistant](https://raw.githubusercontent.com/yousefahmed2004/Model-Gen-X-Smart-AutoML-DL-RAG/main/images/Screenshot%202026-05-24%20031528.png)

The assistant can help the user:

* Understand the dataset
* Configure a training task
* Select the learning type
* Start model training
* Understand training results
* Interact with trained models
* Work with RAG pipelines

Instead of manually configuring every step, the user can describe the desired task using natural language.

---

# 📂 Dataset Upload

Users can upload their datasets directly through the platform.

![Dataset Upload](https://raw.githubusercontent.com/yousefahmed2004/Model-Gen-X-Smart-AutoML-DL-RAG/main/images/Screenshot%202026-05-24%20031636.png)

Supported dataset formats include:

* CSV
* TSV
* Excel
* Parquet
* JSON

After uploading a dataset, the platform can inspect the data and prepare it for the selected machine learning workflow.

---

# 🧠 ML Task Selection

Before training, users can choose the type of machine learning problem they want to solve.

![ML Task Selection](https://raw.githubusercontent.com/yousefahmed2004/Model-Gen-X-Smart-AutoML-DL-RAG/main/images/Screenshot%202026-08-25%20020423.png)

The platform supports different learning tasks, including:

### Classification

Used when the target is categorical.

Examples:

```text
Spam / Not Spam
Disease / No Disease
Customer Churn
Image Class
```

### Regression

Used when the target is a continuous numerical value.

Examples:

```text
House Price
Sales Prediction
Temperature Prediction
Demand Forecasting
```

### Deep Learning

The platform can also route suitable tasks toward Deep Learning workflows.

---

# ⚙️ AutoML Pipeline

The AutoML engine automates the traditional machine learning workflow.

```text
Dataset
   │
   ▼
Data Inspection
   │
   ▼
Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
Task Detection
   │
   ▼
Model Selection
   │
   ▼
Training
   │
   ▼
Cross Validation
   │
   ▼
Evaluation
   │
   ▼
Best Model
   │
   ▼
Save Model
   │
   ▼
Prediction Playground
```

The platform can automatically compare multiple candidate models and select the best-performing model according to the evaluation metrics.

---

# 🏆 Automated Model Selection

The AutoML engine evaluates multiple machine learning algorithms instead of forcing the user to manually select a model.

Candidate models can include:

* Random Forest
* Logistic Regression
* Ridge Regression
* Gradient Boosting
* Decision Tree

The general workflow is:

```text
Candidate Models
      │
      ├── Model 1
      ├── Model 2
      ├── Model 3
      ├── Model 4
      └── Model 5
             │
             ▼
      Cross Validation
             │
             ▼
       Compare Metrics
             │
             ▼
       Select Winner
             │
             ▼
        Train Final
             │
             ▼
        Save Pipeline
```

---

# 📊 Model Evaluation

After training, the platform evaluates the trained model using appropriate metrics.

For classification tasks, metrics can include:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

For regression tasks, metrics can include:

* MAE
* MSE
* RMSE
* R²

The trained model is saved as a portable artifact that can be loaded later for prediction.

---

# 🧪 Machine Learning Playground

After training a Machine Learning model, users can test it directly through the platform.

![Machine Learning Playground](https://raw.githubusercontent.com/yousefahmed2004/Model-Gen-X-Smart-AutoML-DL-RAG/main/images/Screenshot%202026-05-24%20031623.png)

The playground allows users to:

* Enter prediction features
* Run the trained model
* Generate predictions
* Experiment with different inputs
* Test the model without writing code

```text
Trained ML Model
       │
       ▼
   Playground
       │
       ▼
User Input
       │
       ▼
   Prediction
```

---

# 🧬 Deep Learning Playground

Model Gen X also provides an environment for experimenting with Deep Learning models.

![Deep Learning Playground](https://raw.githubusercontent.com/yousefahmed2004/Model-Gen-X-Smart-AutoML-DL-RAG/main/images/Screenshot%202026-06-12%20160931.png)

Users can interact with trained Deep Learning models and test them through the platform.

The Deep Learning workflow is designed to provide a similar experience to the Machine Learning playground while supporting neural-network-based models.

---

# 📚 RAG — Retrieval-Augmented Generation

Model Gen X includes a RAG workflow that allows users to build AI systems based on their own documents and knowledge.

```text
Documents
    │
    ▼
Document Upload
    │
    ▼
Text Extraction
    │
    ▼
Chunking
    │
    ▼
Embeddings
    │
    ▼
Vector Database
    │
    ▼
Retriever
    │
    ▼
LLM
    │
    ▼
AI Response
```

Instead of relying only on the LLM's internal knowledge, the system retrieves relevant information from the user's uploaded documents before generating a response.

---

# 📄 RAG Training & Testing

Users can upload documents and create a knowledge base for the RAG system.

![RAG Training](https://raw.githubusercontent.com/yousefahmed2004/Model-Gen-X-Smart-AutoML-DL-RAG/main/images/Screenshot%202026-05-28%20070235.png)

The workflow allows users to:

1. Upload training documents
2. Process the documents
3. Build the knowledge base
4. Generate embeddings
5. Store the vectors
6. Retrieve relevant information
7. Ask questions
8. Generate context-aware responses

---

# 🧠 AI Agent Architecture

The AI Agent acts as the intelligent interface between the user and the platform.

```text
                       USER
                         │
                         ▼
                  AI Assistant
                         │
                         ▼
                   AI Agent
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        AutoML      Deep Learning       RAG
          │              │              │
          ▼              ▼              ▼
       Training       Training       Knowledge
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Model / RAG App
                         │
                         ▼
                    Playground
```

The goal is to make the platform accessible through natural language rather than requiring users to understand every underlying implementation detail.

---

# 🏗️ System Architecture

```text
┌───────────────────────┐
│        User           │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│    Web Interface      │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│     AI Assistant      │
│      / AI Agent       │
└───────────┬───────────┘
            │
     ┌──────┼───────┐
     ▼      ▼       ▼
┌────────┐ ┌──────┐ ┌─────────┐
│ AutoML │ │  DL  │ │   RAG   │
└───┬────┘ └──┬───┘ └────┬────┘
    │         │          │
    ▼         ▼          ▼
 Models    Models    Vector Store
    │         │          │
    └─────────┼──────────┘
              ▼
       Prediction / Chat
```

---

# 🔄 End-to-End Workflow

```text
1. User creates an account
              ↓
2. User opens the AI Assistant
              ↓
3. User uploads a dataset
              ↓
4. Platform analyzes the dataset
              ↓
5. User selects Classification / Regression / Deep Learning
              ↓
6. AI Agent configures the workflow
              ↓
7. AutoML / DL pipeline starts
              ↓
8. Candidate models are trained
              ↓
9. Models are evaluated
              ↓
10. Best model is selected
              ↓
11. Model is saved
              ↓
12. User opens the Playground
              ↓
13. User tests the trained model
```

For RAG:

```text
1. User uploads documents
              ↓
2. Documents are processed
              ↓
3. Text is chunked
              ↓
4. Embeddings are generated
              ↓
5. Vectors are stored
              ↓
6. User asks a question
              ↓
7. Relevant chunks are retrieved
              ↓
8. LLM generates the answer
```

---

# 🛠️ Tech Stack

| Technology                  | Purpose                                 |
| --------------------------- | --------------------------------------- |
| **Python**                  | Core backend and ML logic               |
| **FastAPI**                 | REST API backend                        |
| **Scikit-learn**            | Machine Learning / AutoML               |
| **PyTorch / Deep Learning** | Neural network workflows                |
| **RAG**                     | Knowledge retrieval and AI applications |
| **LLMs**                    | AI assistant and generation             |
| **SQLAlchemy**              | Database ORM                            |
| **SQLite / PostgreSQL**     | Persistent data                         |
| **Joblib**                  | Model persistence                       |
| **Pandas**                  | Data processing                         |
| **NumPy**                   | Numerical computing                     |
| **HTML / CSS / JavaScript** | Frontend                                |
| **REST APIs**               | Backend communication                   |

---

# 📦 Project Structure

```text
Model-Gen-X/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── auth.py
│   │   │   ├── projects.py
│   │   │   ├── datasets.py
│   │   │   ├── training.py
│   │   │   └── chat.py
│   │   │
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── security.py
│   │   │
│   │   ├── db/
│   │   │   └── database.py
│   │   │
│   │   ├── models/
│   │   ├── schemas/
│   │   │
│   │   ├── ml/
│   │   │   ├── automl_engine.py
│   │   │   └── dataset_loader.py
│   │   │
│   │   ├── services/
│   │   │   ├── gemini_service.py
│   │   │   ├── kaggle_service.py
│   │   │   └── token_service.py
│   │   │
│   │   └── main.py
│   │
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── css/
│   ├── js/
│   ├── pages/
│   └── index.html
│
├── samples/
│   └── iris.csv
│
├── images/
│   ├── Screenshot 2026-05-24 031528.png
│   ├── Screenshot 2026-05-24 031636.png
│   ├── Screenshot 2026-08-25 020423.png
│   ├── Screenshot 2026-05-24 031623.png
│   ├── Screenshot 2026-06-12 160931.png
│   └── Screenshot 2026-05-28 070235.png
│
├── uploads/
├── trained_models/
└── README.md
```

---

# ⚙️ Installation

## Requirements

* Python **3.11+**
* pip
* Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/yousefahmed2004/Model-Gen-X-Smart-AutoML-DL-RAG.git
cd Model-Gen-X-Smart-AutoML-DL-RAG
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

---

## 4. Configure Environment Variables

Create:

```text
backend/.env
```

Example:

```env
DATABASE_URL=sqlite:///./smart_automl.db

GEMINI_API_KEY=your_api_key_here

GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8000/api/auth/google/callback
```

---

# ▶️ Running the Backend

From the `backend` directory:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

---

# 🌐 Running the Frontend

Open another terminal:

```bash
cd frontend
python -m http.server 5500
```

Then open:

```text
http://localhost:5500
```

---

# 🔐 Authentication

Model Gen X supports:

* Email / Password authentication
* JWT authentication
* Google OAuth 2.0

The authentication layer protects user projects, datasets, trained models, and conversations.

---

# 💾 Model Persistence

Trained models are stored as portable `joblib` artifacts.

A saved model bundle can contain:

```text
Model Bundle
│
├── Pipeline
├── Feature Columns
├── Target Column
├── Task Type
└── Class Names
```

This makes it possible to load the trained model later and perform predictions without retraining.

---

# 📡 API

The backend exposes REST APIs for:

* Authentication
* Projects
* Dataset management
* Model training
* Predictions
* Model downloads
* AI conversations
* Chat history

API documentation is automatically generated by FastAPI.

```text
http://localhost:8000/docs
```

---

# 📊 Supported Machine Learning Workflow

### Classification

```text
Dataset
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Candidate Models
   ↓
Cross Validation
   ↓
Best Model
   ↓
Evaluation
   ↓
Prediction
```

### Regression

```text
Dataset
   ↓
Preprocessing
   ↓
Candidate Models
   ↓
Cross Validation
   ↓
Best Model
   ↓
Regression Metrics
   ↓
Prediction
```

### Deep Learning

```text
Dataset
   ↓
Preprocessing
   ↓
Neural Network
   ↓
Training
   ↓
Validation
   ↓
Evaluation
   ↓
Prediction
```

---

# 📚 RAG Workflow

```text
Documents
    │
    ▼
Upload
    │
    ▼
Extract Text
    │
    ▼
Chunk Documents
    │
    ▼
Generate Embeddings
    │
    ▼
Vector Store
    │
    ▼
Retriever
    │
    ▼
LLM
    │
    ▼
Context-Aware Answer
```

---

# 🔮 Future Improvements

Planned improvements include:

* 🚀 Advanced AutoML
* 🧠 Automated Deep Learning architecture selection
* 🎯 Hyperparameter optimization
* 🏆 More ML algorithms
* 📈 Advanced model comparison
* ⚡ GPU training
* ☁️ Cloud training
* 📦 Large dataset processing
* 🔄 Distributed training
* 🧠 Advanced RAG pipelines
* 🔎 Better document retrieval
* 🤖 More specialized AI Agents
* 📊 Advanced analytics
* 🚀 Production deployment
* 🔐 Advanced security
* 👥 Team collaboration
* 🌐 Model APIs
* 📡 Model deployment endpoints

---

# 📸 Screenshots

## AI Assistant

![AI Assistant](https://raw.githubusercontent.com/yousefahmed2004/Model-Gen-X-Smart-AutoML-DL-RAG/main/images/Screenshot%202026-05-24%20031528.png)

---

## Dataset Upload

![Dataset Upload](https://raw.githubusercontent.com/yousefahmed2004/Model-Gen-X-Smart-AutoML-DL-RAG/main/images/Screenshot%202026-05-24%20031636.png)

---

## ML Task Selection

![ML Task Selection](https://raw.githubusercontent.com/yousefahmed2004/Model-Gen-X-Smart-AutoML-DL-RAG/main/images/Screenshot%202026-08-25%20020423.png)

---

## Machine Learning Playground

![Machine Learning Playground](https://raw.githubusercontent.com/yousefahmed2004/Model-Gen-X-Smart-AutoML-DL-RAG/main/images/Screenshot%202026-05-24%20031623.png)

---

## Deep Learning Playground

![Deep Learning Playground](https://raw.githubusercontent.com/yousefahmed2004/Model-Gen-X-Smart-AutoML-DL-RAG/main/images/Screenshot%202026-06-12%20160931.png)

---

## RAG Training

![RAG Training](https://raw.githubusercontent.com/yousefahmed2004/Model-Gen-X-Smart-AutoML-DL-RAG/main/images/Screenshot%202026-05-28%20070235.png)

---

# 📌 Project Status

**Active Development**

Model Gen X is an AI-powered platform that combines:

```text
AI Agents
    +
AutoML
    +
Deep Learning
    +
RAG
    +
Interactive Playground
    =
Smart ML Development Platform
```

The project is designed to simplify the process of building and experimenting with Machine Learning, Deep Learning, and Retrieval-Augmented Generation systems.

---

# 👨‍💻 Author

**Yousef Ahmed**

AI Engineer | Machine Learning | Deep Learning | NLP | AI Automation

---

# ⭐ Model Gen X


```

**ملاحظة مهمة:** الروابط اللي فوق كلها معمولة بصيغة `raw.githubusercontent.com`، وده الشكل الصح لما الصور تكون فعلًا داخل فولدر `images` في الريبو. وبالتالي GitHub هيعرض الصور داخل الـREADME بدل ما يفتحلك صفحة الـblob.
```
