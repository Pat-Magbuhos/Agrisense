from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import subprocess
import io

app = FastAPI()

def mjpeg_stream():
    command = [
        "libcamera-vid",
        "--nopreview",
        "-t", "0",
        "--width", "640",
        "--height", "480",
        "--framerate", "15",
        "--codec", "mjpeg",
        "-o", "-"
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=0)

    try:
        while True:
            frame = process.stdout.read(1024)
            if not frame:
                break
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    finally:
        process.terminate()

@app.get("/live")
def live_stream():
    return StreamingResponse(mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")
