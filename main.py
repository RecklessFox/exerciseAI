from fastapi import FastAPI, Request
import uvicorn
from inference import generate_text
from inference import generate_image
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
import os


app = FastAPI()
templates = Jinja2Templates(directory="")
model_pathGPT = 'modelGPT'

@app.get("/test", response_class=HTMLResponse)
async def hello(request: Request):
    for path, subdirs, files in os.walk("."):
        for name in files:
            print(os.path.join(path, name))

@app.get("/", response_class=HTMLResponse)
async def hello(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/inferText/{sequence}", response_class=HTMLResponse, status_code = 200)
def infer(request: Request, sequence:str):
    max_len = 800
    lyrics = generate_text(model_pathGPT, sequence, max_len)
    #return templates.TemplateResponse("lyrics.html", {"request": request, "lyrics": lyrics})
    return FileResponse("result.txt")

@app.get("/inferImage/{sequence}", response_class=HTMLResponse, status_code = 200)
async def infer(request: Request, sequence: str):
    generate_image(sequence)
    return FileResponse("cover.jpg")


@app.get("/getText", response_class=HTMLResponse, status_code = 200)
async def infer(request: Request):
    return FileResponse("400len.txt")



if __name__ == "__main__":
    uvicorn.run(app, port = 8000, host="0.0.0.0")