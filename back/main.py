from io import BytesIO
from typing import Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import pdf_to_cells, recognize_cells, save_to_docx

# uvicorn main:app --reload

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

origins = [
    "http://localhost:3001", # without / at the end
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def save(bin_io, name):
    with open(name, 'wb') as out:
        out.write(bin_io.getbuffer())


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/parseForm")
async def parse_form(file: UploadFile = File(...)):
    cells = pdf_to_cells(await file.read())
    recognized_cells = recognize_cells(cells)
    file = save_to_docx(recognized_cells)

    with open('demo.docx', 'rb') as f:
        file = BytesIO(f.read())

    return StreamingResponse(file)
