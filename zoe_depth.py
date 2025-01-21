from transformers import pipeline
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

pipe = pipeline(task="depth-estimation", model="Intel/zoedepth-nyu-kitti", device=0)
result = pipe(image)
depth = result["depth"]

depth.show()