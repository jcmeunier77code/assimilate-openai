#!/usr/bin/env python
"""
A command line tool that uses click to generate images using OpenAI.
"""
import click
from oalib.image_gen import generate_image

# build a click command line tool that takes a prompt and returns a generated image
@click.group()
def cli():
    """A command line tool that uses click to generate images using OpenAI."""


@cli.command("generate")
@click.option("--prompt", help="The prompt to use for image generation.")
@click.option("--size", default="1024x1024", help="The size of the image to generate.")
def generate(prompt, size):
    """Generate an image using OpenAI's API."""
    image_url = generate_image(prompt, size)
    print(image_url)


if __name__ == "__main__":
    cli()
