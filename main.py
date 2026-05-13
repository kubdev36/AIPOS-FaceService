import base64
import json
import os
from typing import List, Literal, Optional

import cv2
import face_recognition
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="AI POS Face Service",
    description="Face detection and face descriptor service for AI F&B POS",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.5"))
LIVENESS_SIDE_POSE_THRESHOLD = float(
    os.getenv("LIVENESS_SIDE_POSE_THRESHOLD", "0.12")
)
LIVENESS_STRAIGHT_TOLERANCE = float(
    os.getenv("LIVENESS_STRAIGHT_TOLERANCE", "0.08")
)
SESSION_SELF_MATCH_THRESHOLD = float(
    os.getenv("SESSION_SELF_MATCH_THRESHOLD", "0.55")
)


class CompareRequest(BaseModel):
    descriptor1: List[float]
    descriptor2: List[float]
    threshold: Optional[float] = None


class CompareManyRequest(BaseModel):
    descriptor: List[float]
    candidates: List[dict]
    threshold: Optional[float] = None


PoseLabel = Literal["straight", "left", "right"]


def read_image_from_upload(file_bytes: bytes):
    np_array = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb


def validate_descriptor(descriptor: List[float], name: str = "descriptor"):
    if not isinstance(descriptor, list):
        raise HTTPException(status_code=400, detail=f"{name} must be a list")

    if len(descriptor) != 128:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must have 128 numbers, got {len(descriptor)}",
        )

    try:
        return np.array(descriptor, dtype=np.float64)
    except Exception:
        raise HTTPException(status_code=400, detail=f"{name} contains invalid values")


def decode_base64_image(image_base64: str) -> bytes:
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        return base64.b64decode(image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")


def read_image_from_base64(image_base64: str):
    return read_image_from_upload(decode_base64_image(image_base64))


def detect_single_face(image_rgb):
    face_locations = face_recognition.face_locations(
        image_rgb,
        model="hog"
    )

    if len(face_locations) == 0:
        raise HTTPException(status_code=400, detail="No face detected")

    if len(face_locations) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected")

    return face_locations


def extract_face_descriptor_from_image(image_rgb):
    face_locations = detect_single_face(image_rgb)

    face_encodings = face_recognition.face_encodings(
        image_rgb,
        known_face_locations=face_locations
    )

    if len(face_encodings) == 0:
        raise HTTPException(status_code=400, detail="Cannot extract face descriptor")

    descriptor = face_encodings[0]

    top, right, bottom, left = face_locations[0]

    return {
        "descriptor": descriptor.tolist(),
        "face_location": {
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
            "left": int(left),
        },
    }


def estimate_face_pose(image_rgb, face_locations) -> dict:
    landmarks_list = face_recognition.face_landmarks(
        image_rgb,
        face_locations=face_locations,
        model="small"
    )

    if not landmarks_list:
        raise HTTPException(status_code=400, detail="Cannot estimate face pose")

    landmarks = landmarks_list[0]
    left_eye = np.array(landmarks["left_eye"], dtype=np.float64)
    right_eye = np.array(landmarks["right_eye"], dtype=np.float64)
    nose_tip = np.array(landmarks["nose_tip"][0], dtype=np.float64)

    left_eye_center = left_eye.mean(axis=0)
    right_eye_center = right_eye.mean(axis=0)
    eye_midpoint = (left_eye_center + right_eye_center) / 2.0
    eye_distance = float(np.linalg.norm(right_eye_center - left_eye_center))

    if eye_distance == 0:
        raise HTTPException(status_code=400, detail="Cannot estimate face pose")

    nose_offset_ratio = float((nose_tip[0] - eye_midpoint[0]) / eye_distance)

    if abs(nose_offset_ratio) <= LIVENESS_STRAIGHT_TOLERANCE:
        pose = "straight"
    elif nose_offset_ratio < 0:
        pose = "left"
    else:
        pose = "right"

    return {
        "pose": pose,
        "nose_offset_ratio": nose_offset_ratio,
        "straight_tolerance": LIVENESS_STRAIGHT_TOLERANCE,
        "side_pose_threshold": LIVENESS_SIDE_POSE_THRESHOLD,
    }


def validate_expected_pose(
    detected_pose: str,
    pose_metrics: dict,
    expected_pose: PoseLabel,
):
    signed_nose_offset_ratio = float(pose_metrics["nose_offset_ratio"])
    nose_offset_ratio = abs(signed_nose_offset_ratio)

    if expected_pose == "straight":
        if detected_pose != "straight":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Expected a straight face image "
                    f"(detected={detected_pose}, "
                    f"nose_offset_ratio={signed_nose_offset_ratio:.3f})"
                )
            )
        return

    if detected_pose != expected_pose or nose_offset_ratio < LIVENESS_SIDE_POSE_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected face pose: {expected_pose} "
                f"(detected={detected_pose}, "
                f"nose_offset_ratio={signed_nose_offset_ratio:.3f}, "
                f"threshold={LIVENESS_SIDE_POSE_THRESHOLD:.3f})"
            )
        )


def extract_face_sample(
    image_rgb,
    expected_pose: Optional[PoseLabel] = None,
):
    face_locations = detect_single_face(image_rgb)
    result = extract_face_descriptor_from_image(image_rgb)
    pose_metrics = estimate_face_pose(image_rgb, face_locations)

    if expected_pose is not None:
        validate_expected_pose(
            detected_pose=pose_metrics["pose"],
            pose_metrics=pose_metrics,
            expected_pose=expected_pose,
        )

    return {
        "descriptor": result["descriptor"],
        "face_location": result["face_location"],
        "pose": pose_metrics["pose"],
        "pose_metrics": pose_metrics,
    }


def average_descriptors(descriptors: List[List[float]]) -> List[float]:
    descriptor_arrays = [
        validate_descriptor(descriptor, f"descriptors[{index}]")
        for index, descriptor in enumerate(descriptors)
    ]
    averaged = np.mean(np.stack(descriptor_arrays), axis=0)
    return averaged.astype(np.float64).tolist()


def ensure_same_person(descriptors: List[List[float]]) -> List[dict]:
    if len(descriptors) <= 1:
        return []

    reference_descriptor = validate_descriptor(descriptors[0], "descriptors[0]")
    comparisons = []

    for index in range(1, len(descriptors)):
        current_descriptor = validate_descriptor(
            descriptors[index],
            f"descriptors[{index}]"
        )
        distance = float(np.linalg.norm(reference_descriptor - current_descriptor))

        if distance > SESSION_SELF_MATCH_THRESHOLD:
            raise HTTPException(
                status_code=400,
                detail="Face session failed consistency check"
            )

        comparisons.append(
            {
                "from_index": 0,
                "to_index": index,
                "distance": distance,
                "threshold": SESSION_SELF_MATCH_THRESHOLD,
            }
        )

    return comparisons


async def read_uploaded_image(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    file_bytes = await file.read()
    return read_image_from_upload(file_bytes)


def calculate_distance(descriptor1: List[float], descriptor2: List[float]):
    a = validate_descriptor(descriptor1, "descriptor1")
    b = validate_descriptor(descriptor2, "descriptor2")

    distance = np.linalg.norm(a - b)

    return float(distance)


@app.get("/")
def root():
    return {
        "success": True,
        "message": "AI POS Face Service is running",
        "threshold": FACE_MATCH_THRESHOLD,
    }


@app.get("/health")
def health_check():
    return {
        "success": True,
        "status": "healthy",
    }


@app.post("/face/descriptor")
async def get_face_descriptor(file: UploadFile = File(...)):
    image_rgb = await read_uploaded_image(file)
    result = extract_face_sample(image_rgb, expected_pose="straight")

    return {
        "success": True,
        "message": "Face descriptor extracted successfully",
        "descriptor": result["descriptor"],
        "face_location": result["face_location"],
    }


@app.post("/face/descriptor-multi")
async def get_face_descriptors_multi(
    straight_file: UploadFile = File(...),
    left_file: UploadFile = File(...),
    right_file: UploadFile = File(...),
):
    samples = [
        ("straight", await read_uploaded_image(straight_file)),
        ("left", await read_uploaded_image(left_file)),
        ("right", await read_uploaded_image(right_file)),
    ]

    extracted_samples = []

    for expected_pose, image_rgb in samples:
        extracted_samples.append(
            extract_face_sample(image_rgb, expected_pose=expected_pose)
        )

    descriptors = [sample["descriptor"] for sample in extracted_samples]
    averaged_descriptor = average_descriptors(descriptors)

    return {
        "success": True,
        "message": "Face descriptors extracted successfully",
        "threshold": FACE_MATCH_THRESHOLD,
        "samples": extracted_samples,
        "descriptors": descriptors,
        "average_descriptor": averaged_descriptor,
        "recommended_storage": (
            "Store all three descriptors as separate rows per user "
            "and optionally keep the average descriptor for fallback"
        ),
    }


@app.post("/face/compare")
async def compare_faces(payload: CompareRequest):
    threshold = payload.threshold or FACE_MATCH_THRESHOLD

    distance = calculate_distance(payload.descriptor1, payload.descriptor2)

    matched = distance < threshold

    return {
        "success": True,
        "matched": matched,
        "distance": distance,
        "threshold": threshold,
    }


@app.post("/face/compare-user")
async def compare_user_faces(payload: CompareManyRequest):
    threshold = payload.threshold or FACE_MATCH_THRESHOLD
    input_descriptor = validate_descriptor(payload.descriptor, "descriptor")

    matches = []

    for index, candidate in enumerate(payload.candidates):
        candidate_descriptor = candidate.get("descriptor")

        if not candidate_descriptor:
            continue

        try:
            saved_descriptor = validate_descriptor(
                candidate_descriptor,
                f"candidates[{index}].descriptor"
            )
        except HTTPException:
            continue

        distance = float(np.linalg.norm(input_descriptor - saved_descriptor))
        matches.append(
            {
                "id": candidate.get("id"),
                "user_id": candidate.get("user_id"),
                "distance": distance,
                "matched": distance < threshold,
            }
        )

    if not matches:
        return {
            "success": True,
            "matched": False,
            "message": "No valid candidate found",
            "threshold": threshold,
            "best_match": None,
            "matches": [],
        }

    matches.sort(key=lambda item: item["distance"])
    best_match = matches[0]

    return {
        "success": True,
        "matched": any(item["matched"] for item in matches),
        "threshold": threshold,
        "best_match": best_match,
        "matches": matches,
    }


@app.post("/face/compare-many")
async def compare_many_faces(payload: CompareManyRequest):
    threshold = payload.threshold or FACE_MATCH_THRESHOLD

    input_descriptor = validate_descriptor(payload.descriptor, "descriptor")

    best_match = None
    best_distance = float("inf")

    for candidate in payload.candidates:
        candidate_id = candidate.get("id")
        candidate_user_id = candidate.get("user_id")
        candidate_descriptor = candidate.get("descriptor")

        if not candidate_descriptor:
            continue

        try:
            saved_descriptor = validate_descriptor(
                candidate_descriptor,
                "candidate_descriptor"
            )

            distance = float(np.linalg.norm(input_descriptor - saved_descriptor))

            if distance < best_distance:
                best_distance = distance
                best_match = {
                    "id": candidate_id,
                    "user_id": candidate_user_id,
                    "distance": distance,
                }

        except Exception:
            continue

    if best_match is None:
        return {
            "success": True,
            "matched": False,
            "message": "No valid candidate found",
            "best_match": None,
            "threshold": threshold,
        }

    matched = best_distance < threshold

    return {
        "success": True,
        "matched": matched,
        "best_match": best_match,
        "best_distance": best_distance,
        "threshold": threshold,
    }


@app.post("/face/descriptor-from-base64")
async def get_face_descriptor_from_base64(image_base64: str = Form(...)):
    image_rgb = read_image_from_base64(image_base64)
    result = extract_face_sample(image_rgb)

    return {
        "success": True,
        "message": "Face descriptor extracted successfully",
        "descriptor": result["descriptor"],
        "face_location": result["face_location"],
    }


@app.post("/face/liveness-check")
async def liveness_check(
    straight_file: UploadFile = File(...),
    left_file: UploadFile = File(...),
    right_file: UploadFile = File(...),
):
    steps = [
        ("straight", await read_uploaded_image(straight_file)),
        ("left", await read_uploaded_image(left_file)),
        ("right", await read_uploaded_image(right_file)),
    ]

    results = []

    for expected_pose, image_rgb in steps:
        results.append(extract_face_sample(image_rgb, expected_pose=expected_pose))

    return {
        "success": True,
        "message": "Simple liveness check passed",
        "liveness_passed": True,
        "sequence": ["straight", "left", "right"],
        "steps": results,
    }


@app.post("/face/challenge-verify")
async def verify_face_challenge_session(
    files: List[UploadFile] = File(...),
    challenge_types: str = Form(...),
):
    try:
        parsed_challenge_types = json.loads(challenge_types)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid challenge_types payload")

    if not isinstance(parsed_challenge_types, list) or not parsed_challenge_types:
        raise HTTPException(
            status_code=400,
            detail="challenge_types must be a non-empty array"
        )

    if len(files) != len(parsed_challenge_types):
        raise HTTPException(
            status_code=400,
            detail="Number of uploaded files must match challenge_types"
        )

    steps = []
    descriptors = []

    for index, expected_pose in enumerate(parsed_challenge_types):
        if expected_pose != "straight":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported challenge type: {expected_pose}"
            )

        image_rgb = await read_uploaded_image(files[index])
        sample = extract_face_sample(image_rgb, expected_pose=expected_pose)
        steps.append(
            {
                "index": index,
                "expected_pose": expected_pose,
                "detected_pose": sample["pose"],
                "face_location": sample["face_location"],
                "pose_metrics": sample["pose_metrics"],
            }
        )
        descriptors.append(sample["descriptor"])

    comparisons = ensure_same_person(descriptors)
    average_descriptor = average_descriptors(descriptors)

    return {
        "success": True,
        "message": "Face challenge session verified",
        "session_passed": True,
        "challenge_types": parsed_challenge_types,
        "steps": steps,
        "descriptors": descriptors,
        "average_descriptor": average_descriptor,
        "consistency_checks": comparisons,
        "threshold": FACE_MATCH_THRESHOLD,
        "session_self_match_threshold": SESSION_SELF_MATCH_THRESHOLD,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
