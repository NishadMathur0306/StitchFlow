# StitchFlow
An AI-powered photo stitching web app that intelligently merges multiple images into a seamless panorama. It supports folder uploads, smart image sorting, multi-output previews, and PDF export of results.
# AI Photo Stitching Web App

This project is an AI-powered web application that allows users to seamlessly stitch multiple images together into a panoramic view. The app uses computer vision techniques for stitching, supports folder-based uploads, and includes functionality for smart sorting, live previews, PDF export, and downloads of the stitched output.

## 🚀 Features

- 📁 Upload multiple images or entire folders
- 🧠 AI-assisted stitching based on user descriptions
- 🧵 Poisson blending and homography-based stitching
- 🔍 Preview stitched output directly on the website
- ✅ Sort images into "good" and "bad" categories
- 📄 Export stitched images as a downloadable PDF
- ⬇️ Download individual stitched results

## 🛠️ Tech Stack

- **Backend:** FastAPI
- **Frontend:** HTML, Jinja2 Templates
- **Image Processing:** OpenCV
- **PDF Export:** FPDF
- **Templating & Static:** Jinja2, StaticFiles

## 📦 Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-photo-stitcher.git
   cd ai-photo-stitcher
2. Create a virtual environment and install dependencies
   python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Run the server
  uvicorn backend.main:app --reload
