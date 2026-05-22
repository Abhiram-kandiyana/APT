from main import load_files
import os

def test_load():
    with open("test.txt", "w") as f:
        f.write("Hello")
    
    content = load_files("test.txt")
    assert content == "Hello"
    print("load_files passed")
    os.remove("test.txt")

if __name__ == "__main__":
    test_load()
