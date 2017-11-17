import os
root = os.path.dirname( __file__)
LogPath = os.path.join(root, "ocr.log")

def WriteLog(message):
    with open(LogPath, "a") as f:
        f.write(message + "\n")