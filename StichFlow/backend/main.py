import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List
from fpdf import FPDF
import uuid
import shutil
import logging

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Setup FastAPI and folders
app = FastAPI()

STATIC_DIR = "backend/static"
TEMPLATES_DIR = "backend/templates"
STITCHED_DIR = os.path.join(STATIC_DIR, "stitched")
UPLOAD_DIR = os.path.join(STATIC_DIR, "to_stitch")
SORTED_GOOD = os.path.join(STATIC_DIR, "sorted/good")
SORTED_BAD = os.path.join(STATIC_DIR, "sorted/bad")

os.makedirs(STITCHED_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SORTED_GOOD, exist_ok=True)
os.makedirs(SORTED_BAD, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Stitching logic
def detect_and_match_keypoints(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches

def calculate_homography(kp1, kp2, matches):
    if len(matches) < 4:
        raise ValueError("Not enough matches found between images.")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return M

def poisson_blending(image1, image2, mask):
    center = (image2.shape[1] // 2, image2.shape[0] // 2)
    return cv2.seamlessClone(image1, image2, mask, center, cv2.NORMAL_CLONE)

def stitch_images(image1, image2):
    kp1, kp2, matches = detect_and_match_keypoints(image1, image2)
    M = calculate_homography(kp1, kp2, matches)
    h, w = image1.shape[:2]
    warped_image2 = cv2.warpPerspective(image2, M, (w, h))
    mask = cv2.cvtColor(warped_image2, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    mask = cv2.merge([mask, mask, mask])
    return poisson_blending(image1, warped_image2, mask)

def stitch_images_with_ai(image_paths: List[str], description: str) -> str:
    images = [cv2.imread(path) for path in image_paths if cv2.imread(path) is not None]
    if len(images) < 2:
        return None
    stitched_img = images[0]
    for next_img in images[1:]:
        try:
            stitched_img = stitch_images(stitched_img, next_img)
        except Exception as e:
            logging.error(f"Error stitching: {e}")
            continue
    output_id = str(uuid.uuid4())[:8]
    output_filename = f"stitched_{output_id}.jpg"
    output_path = os.path.join(STITCHED_DIR, output_filename)
    cv2.imwrite(output_path, stitched_img)
    return f"/static/stitched/{output_filename}"

def create_pdf(image_paths: List[str], output_pdf_path: str):
    pdf = FPDF()
    for img_path in image_paths:
        pdf.add_page()
        pdf.image(img_path, x=10, y=10, w=180)
    pdf.output(output_pdf_path)

# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    stitched_images = [
        f"/static/stitched/{f}" for f in os.listdir(STITCHED_DIR) if f.endswith(".jpg")
    ]
    return templates.TemplateResponse("form.html", {
        "request": request,
        "stitched_images": stitched_images
    })

@app.post("/stitch/", response_class=HTMLResponse)
async def stitch(
    request: Request,
    description: str = Form(...),
    files: List[UploadFile] = File(...)
):
    image_paths = []
    for file in files:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        cv2.imwrite(save_path, img)
        image_paths.append(save_path)

    stitched_url = stitch_images_with_ai(image_paths, description)
    stitched_images = [
        f"/static/stitched/{f}" for f in os.listdir(STITCHED_DIR) if f.endswith(".jpg")
    ]
    return templates.TemplateResponse("form.html", {
        "request": request,
        "message": "Stitching completed!",
        "stitched_images": stitched_images
    })

@app.post("/sort/")
async def sort_images(
    folder: List[UploadFile] = File(...),
    good_folder: str = Form(...),
    bad_folder: str = Form(...)
):
    for file in folder:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        # Dummy logic: Use size as quality proxy
        if img.shape[0] > 300 and img.shape[1] > 300:
            save_path = os.path.join(SORTED_GOOD, file.filename)
        else:
            save_path = os.path.join(SORTED_BAD, file.filename)
        cv2.imwrite(save_path, img)
    return {"message": "Images sorted successfully"}

@app.get("/download/{image_name}")
async def download_image(image_name: str):
    image_path = os.path.join(STITCHED_DIR, image_name)
    return FileResponse(image_path, media_type='image/jpeg', filename=image_name)
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    print("🚀 Homepage loaded")
    ...


@app.get("/export-pdf")
async def export_pdf():
    image_files = [os.path.join(STITCHED_DIR, f) for f in os.listdir(STITCHED_DIR) if f.endswith(".jpg")]
    pdf_path = os.path.join(STITCHED_DIR, "stitched_output.pdf")
    create_pdf(image_files, pdf_path)
    return FileResponse(pdf_path, media_type='application/pdf', filename="stitched_output.pdf")
