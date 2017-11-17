
# import os
path = "test"
# path = os.path.join(os.path.dirname(__file__), "test.log")
with open("test.log", "a") as f:
    f.write(path + "\n")