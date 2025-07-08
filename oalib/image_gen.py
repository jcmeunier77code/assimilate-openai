"""A library for generating images using OpenAI."""

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

# build a function that takes a prompt and returns a generated image
def generate_image(prompt, size="1024x1024"):
    """Generate an image using OpenAI's API."""
    response = client.images.generate(  # pylint: disable=no-member
        prompt=prompt,
        n=1,
        size=size,
    )   
    image_url = response.data[0].url
    return image_url
