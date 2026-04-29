# Public Backend Deployment

This backend can be hosted publicly now, but it should be deployed as a Docker web service, not as a serverless function.

## Why this deployment shape

- The API loads TensorFlow and a trained Keras model at process startup.
- The server only needs the backend code, configs, and the `latest` production model files.
- The training datasets are not needed on the public server and are excluded by `.dockerignore`.

## Files the server uses

- `app/backend/`
- `src/`
- `configs/train_strong_cnn_pneumonia_focus.yaml`
- `models_strong_cnn_pneumonia_focus/latest/best_model.keras`
- `models_strong_cnn_pneumonia_focus/latest/class_names.json`
- `requirements.txt`
- `Dockerfile`

## Deploy steps

1. Push `/Users/diyarao/Documents/Playground/respiratory_ai_rebuild` to a Git repo or point your hosting provider at this folder as the app root.
2. Create a new Docker-based web service on your hosting provider.
3. Let the provider build from the included `Dockerfile`.
4. Set the health check path to `/ready`.
5. Wait for the service to report ready, then copy the public `https://...` URL.

## Important endpoints

- `/health`: process is up and reports config/model metadata.
- `/ready`: returns `200` only when the TensorFlow model is loaded and ready to serve predictions.
- `/predict`: multipart upload endpoint used by the Android app.

Use `/ready` for platform health checks. `/health` is informative, but it can still return `200` while the model is missing.

## Android app wiring

Once the service is live, build the Android QA or release APK with the public URL baked in:

```bash
cd /Users/diyarao/Documents/Playground/respiratory_ai_android_app
./gradlew assembleQa -PdeviceApiBaseUrl=https://your-public-backend.example.com/
```

You can also open the QA app's Connection section and paste the public URL manually for quick testing.

## Practical notes

- If the service boots but never becomes ready, the host likely does not have enough memory for TensorFlow model startup.
- `RESP_AI_PRELOAD_MODEL=1` is already enabled in the container so startup failures surface at deploy time.
- If you ever want to switch models, update `RESP_AI_MODEL` or replace the files under `models_strong_cnn_pneumonia_focus/latest/`.
