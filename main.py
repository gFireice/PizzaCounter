from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


from datetime import datetime
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sqlite3
import cv2
import json
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from openpyxl import Workbook

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    print("Загрузка модели...")
    module = hub.load('https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1')
    app.state.model = module.signatures['default']
    print("Модель загружена.")

    print("Инициализация базы данных...")
    conn = sqlite3.connect('history.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            result TEXT
        )
    ''')
    conn.commit()
    app.state.conn = conn
    app.state.cursor = cursor
    print("База данных готова.")


def detect_objects(image_np, model):
    converted_img = tf.image.convert_image_dtype(image_np, tf.float32)[tf.newaxis, ...]
    result = model(converted_img)
    result = {k: v.numpy() for k, v in result.items()}

    count = 0
    image_np = image_np.copy()
    for score, label, box in zip(result["detection_scores"],
                                  result["detection_class_entities"],
                                  result["detection_boxes"]):
        if score > 0.5 and label.decode() == "Pizza":
            ymin, xmin, ymax, xmax = box
            h, w, _ = image_np.shape
            pt1 = (int(xmin * w), int(ymin * h))
            pt2 = (int(xmax * w), int(ymax * h))
            cv2.rectangle(image_np, pt1, pt2, (0, 255, 0), 2)
            count += 1
    return image_np, count


@app.post("/process")
async def process_image(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(pil_image)

    processed_image, count = detect_objects(image_np, app.state.model)

    result_path = "static/result.jpg"
    cv2.imwrite(result_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))


    app.state.cursor.execute(
        "INSERT INTO requests (timestamp, result) VALUES (?, ?)",
        (datetime.now().isoformat(), json.dumps({"count": count}))
    )
    app.state.conn.commit()

    return JSONResponse(content={"count": count})

@app.get("/report/pdf")
async def generate_pdf():
    cursor = app.state.cursor
    cursor.execute("SELECT timestamp, result FROM requests")
    records = cursor.fetchall()

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica", 14)
    c.drawString(50, height - 50, "Report on processed images")

    c.setFont("Helvetica", 11)
    y = height - 80
    for ts, res in records:
        count = json.loads(res)['count']
        c.drawString(50, y, f"{ts} — Pizzas found: {count}")
        y -= 20
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = height - 50

    c.save()
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=report.pdf"
    })

@app.get("/report/excel")
async def generate_excel():
    cursor = app.state.cursor
    cursor.execute("SELECT timestamp, result FROM requests")
    records = cursor.fetchall()

    wb = Workbook()
    ws = wb.active
    ws.title = "Отчёт"
    ws.append(["Дата и время", "Количество пицц"])

    for ts, res in records:
        count = json.loads(res)['count']
        ws.append([ts, count])

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={
        "Content-Disposition": "attachment; filename=report.xlsx"
    })

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")