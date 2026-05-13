# AI POS Face Service

Python FastAPI service for face detection, face descriptor extraction, multi-sample enrollment, and simple liveness validation.

## Run local

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

## Environment

```env
PORT=8000
FACE_MATCH_THRESHOLD=0.5
LIVENESS_SIDE_POSE_THRESHOLD=0.12
LIVENESS_STRAIGHT_TOLERANCE=0.08
```

## Recommended flow

### 1. Register with 3 images

Capture 3 images for each user:

- `straight`: look straight at the camera
- `left`: turn slightly left
- `right`: turn slightly right

Call `POST /face/descriptor-multi` with 3 files:

- `straight_file`
- `left_file`
- `right_file`

Response returns:

- `samples`: full info for each capture
- `descriptors`: the 3 raw descriptors
- `average_descriptor`: the mean descriptor across the 3 captures

Recommended storage:

- store all 3 descriptors as separate rows for the same `user_id`
- optionally keep `average_descriptor` as a fallback

Example database rows:

```text
user_id=1, descriptor=sample_1
user_id=1, descriptor=sample_2
user_id=1, descriptor=sample_3
```

### 2. Login by comparing against many descriptors

For login, send the current face descriptor and all saved descriptors of that user to:

- `POST /face/compare-user`

If at least one stored descriptor matches under the threshold, login can pass.

Use:

- `0.45 - 0.5` for stricter matching
- `0.55` if the system rejects too often

Default is now:

```env
FACE_MATCH_THRESHOLD=0.5
```

### 3. Simple liveness check

Call `POST /face/liveness-check` with:

- `straight_file`
- `left_file`
- `right_file`

This validates the user follows the expected pose sequence:

1. straight
2. left
3. right

This is a basic anti-photo step using face pose heuristics. It is helpful, but it is not strong anti-spoofing like blink detection, video challenge-response, or depth-based liveness.

## Endpoints

- `GET /` health summary and current threshold
- `GET /health` service health
- `POST /face/descriptor` extract a descriptor from one image file
- `POST /face/descriptor-from-base64` extract a descriptor from one base64 image
- `POST /face/descriptor-multi` extract 3 descriptors plus one averaged descriptor
- `POST /face/compare` compare 2 descriptors directly
- `POST /face/compare-user` compare 1 input descriptor against many stored descriptors for one user
- `POST /face/compare-many` compare 1 input descriptor against many candidate descriptors
- `POST /face/liveness-check` validate the `straight -> left -> right` capture flow
