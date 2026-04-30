---
title: Respiratory AI Backend
emoji: "🫁"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8080
---

# Respiratory AI Backend

This repository contains the deployable backend for the Respiratory AI mobile app.

It includes:

- Flask inference API
- TensorFlow/Keras respiratory disease model
- Render-ready Docker configuration
- Hugging Face Docker Space compatibility
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

## Hugging Face Spaces deployment

This repo is also prepared for Hugging Face as a Docker Space.

- SDK: `docker`
- App port: `8080`
- Public endpoints remain:
  - `/`
  - `/health`
  - `/ready`
  - `/predict`

When deployed as a Space, opening the Space URL will show a simple JSON landing response instead of a `404`.

## Required endpoints

- `/health`
- `/ready`
- `/predict`
