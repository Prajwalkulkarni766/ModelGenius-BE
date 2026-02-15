# ModelGenius Backend

Backend services for the ModelGenius ML platform.

## Architecture

The backend consists of two services:

1. **Node.js API Server** - Express.js REST API
2. **Python ML Service** - FastAPI for model training

## Tech Stack

### Node.js Backend
- **Runtime**: Node.js
- **Framework**: Express.js
- **Database**: MongoDB (Mongoose)
- **Auth**: JWT + bcrypt
- **File Upload**: Multer

### Python ML Service
- **Framework**: FastAPI
- **ML Libraries**: Scikit-learn, NumPy, Pandas
- **Serialization**: Joblib

## Prerequisites

- Node.js 18+
- Python 3.11+
- MongoDB

## Installation

### Node.js Backend
```bash
cd node-backend
npm install
```

### Python Backend
```bash
cd python-backend
pip install -r requirements.txt
```

## Running the Backend

### Node.js Server (Port 5000)
```bash
cd node-backend
npm run dev
```

### Python ML Service (Port 8000)
```bash
cd python-backend
python main.py
# or
uvicorn main:app --reload --port 8000
```

## Environment Variables

Create `.env` files in respective directories and dump all the required values from `.env.example` file to `.env`.

## API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `POST /api/auth/forgot-password` - Request password reset
- `POST /api/auth/reset-password/:token` - Reset password

### Projects
- `GET /api/projects` - List user projects
- `POST /api/projects` - Create project
- `GET /api/projects/:id` - Get project details
- `PUT /api/projects/:id` - Update project
- `DELETE /api/projects/:id` - Delete project

### Models
- `GET /api/models` - List models
- `POST /api/models` - Create model
- `GET /api/models/:id` - Get model details
- `DELETE /api/models/:id` - Delete model
- `POST /api/models/:id/train` - Train model
- `POST /api/models/:id/predict` - Make predictions

### Datasets
- `GET /api/datasets` - List datasets
- `POST /api/datasets` - Upload dataset
- `GET /api/datasets/:id` - Get dataset info
- `DELETE /api/datasets/:id` - Delete dataset

### Settings
- `GET /api/settings` - Get user settings
- `PUT /api/settings` - Update settings

## ML Training API (Python)

### Endpoints
- `POST /train` - Train a model
- `POST /predict` - Make predictions
- `GET /health` - Health check

### Supported Algorithms
- Classification: Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting
- Regression: Linear Regression, Random Forest, Gradient Boosting

### Supported Preprocessing
- Encoding: One-Hot, Label, Ordinal, Binary, Frequency, Target, Hashing
- Normalization: Z-Score, Min-Max, Robust, MaxAbs, Log, Power Transform, Quantile
- Cleaning: Drop rows/columns, Mean, Median, Mode, Forward/Backward fill, KNN, Interpolation

## Project Structure

```
ModelGenius-BE/
├── node-backend/
│   ├── src/
│   │   ├── config/         # Configuration
│   │   ├── controllers/    # Route controllers
│   │   ├── middleware/     # Express middleware
│   │   ├── models/         # Mongoose models
│   │   ├── routes/         # API routes
│   │   ├── utils/          # Utility functions
│   │   └── index.js        # Entry point
│   └── public/             # Static files (uploads)
│
└── python-backend/
    ├── models/             # Saved ML models
    ├── main.py             # FastAPI entry point
    ├── training.py         # Training logic
    ├── preprocessing.py   # Data preprocessing
    ├── code_generator.py  # Model code generation
    └── requirements.txt   # Python dependencies
```
