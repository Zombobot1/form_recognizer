from io import BytesIO
from typing import Optional
from pathy import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


from app import pdf_to_cells, recognize_cells, save_to_docx, write_bytesio_to_file


# uvicorn main:app --reload

app = FastAPI()
pwd = os.getcwd()+"/back"
# os.chdir(pwd)
app.mount("/static", StaticFiles(directory="static"), name="static")

origins = [
    "http://localhost:3001",  # without / at the end
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
    print("starting saving ")

    file = save_to_docx(recognized_cells)

    print("got processed the entire file ")
    # write_bytesio_to_file('copy.docx', file)
    # with open('copy.docx', 'rb') as f:
    #     ali = BytesIO(f.read())

    return StreamingResponse(file)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
