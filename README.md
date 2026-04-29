# Respiratory AI Render Backend

This repository contains the deployable backend for the Respiratory AI mobile app.

It includes:

- Flask inference API
- TensorFlow/Keras respiratory disease model
- Render-ready Docker configuration
- `/ready` health endpoint for deployment checks

## Local run

```bash
pip install -r requirements.txt
PYTHONPATH=src python app/backend/main.py
```

The backend starts on `http://127.0.0.1:8080`.

## Render deployment

This repo is prepared for Render as a Docker web service.

- Runtime: `Docker`
- Health check path: `/ready`
- Port: provided by Render through `$PORT`

If you use the included `render.yaml`, start with the free plan for testing. If the model fails to load during startup, move the service to a larger paid instance in Render.

## Required endpoints

- `/health`
- `/ready`
- `/predict`
