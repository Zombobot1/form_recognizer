from io import BytesIO
from typing import Optional
from pathy import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os


from app import extract_boxes, recognize_cells, save_to_docx, write_bytesio_to_file, recognize_cells_default

os.environ['MODEL'] = 'Model_4_2401+Model_3_2209+Model_deu2+model3+model4'
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
    backend = os.getenv('BACKEND')

    cells = extract_boxes(await file.read())
    recognized_cells = None
    if backend != "simple":

        recognized_cells = recognize_cells(cells)
    else:
        print("using simple back end ")
        recognized_cells = recognize_cells_default(cells)

    file = save_to_docx(recognized_cells)

    print("got processed the entire file ")
    # write_bytesio_to_file('copy.docx', file)
    # with open('copy.docx', 'rb') as f:
    #     ali = BytesIO(f.read())

    return StreamingResponse(file)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
