import base64
import io
from typing import Iterable

from PIL import Image


def image_to_base64(image: Image.Image, *, format: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def base64_to_image(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def images_to_base64(images: Iterable[Image.Image], *, format: str = "PNG") -> list[str]:
    return [image_to_base64(image, format=format) for image in images]
