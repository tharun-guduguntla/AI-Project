import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.views.textsummarization_view import router

print("Here we are",os.getcwd())
app = FastAPI()

# Mount static files (i.e. we are reading DE file)
app.mount("/data", StaticFiles(directory="data"), name="data")

# Including the router
app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Assist Genie!"}