import threading
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main_testcode import simulate_radar_data
from routers import detections
import uvicorn
import testData as td


def create_app():
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
        ],  # Who can call your API?
        allow_methods=["*"],  # What HTTP methods are allowed?
        allow_headers=["*"],  # What headers can be sent?
        # "*" means anyone
    )

    app.include_router(detections.router, tags=["Detections"])

    @app.on_event("startup")
    # thread is used to run the simulation in background (dummy data, only when run standalone)
    def start_simulation():
        if os.environ.get("REAL_DATA_MODE") == "1":
            return  # Real data is provided by main.py; skip simulation
        thread = threading.Thread(
            target=simulate_radar_data, args=(td.Test_radar_data,), daemon=True
        )
        thread.start()

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    app = create_app()
