from PIL import Image, ImageDraw


def create_composite(
    hairball_path="kg_hairball.jpg",
    zoom_path="kg_zoom.jpg",
    output_path="kg_composite.jpg",
    border_color=(120, 0, 180),    # purple
    border_width=10,
    zoom_scale=0.7,
):
    # Load images
    hb = Image.open(hairball_path)
    zm = Image.open(zoom_path)

    # Optional zoom rescale
    if zoom_scale != 1.0:
        zm = zm.resize(
            (int(zm.width * zoom_scale), int(zm.height * zoom_scale)),
            Image.LANCZOS,
        )

    # Canvas side-by-side
    gap = 80
    total_width = hb.width + gap + zm.width
    max_height = max(hb.height, zm.height)

    canvas = Image.new("RGB", (total_width, max_height), "white")

    # Paste hairball on left
    hb_x, hb_y = 0, (max_height - hb.height) // 2
    canvas.paste(hb, (hb_x, hb_y))

    # Paste zoom on right
    right_padding = 250   # <<--- space from the right edge of canvas
    zoom_x = total_width - zm.width - right_padding
    zoom_y = (max_height - zm.height) // 2
    canvas.paste(zm, (zoom_x, zoom_y))

    draw = ImageDraw.Draw(canvas)

    # ---- 1) Purple box on hairball (approx. zoom source region) ----
    # Here we pick a central-ish region in the hairball image.
    # You can tweak these fractions if you want the box a bit higher/lower.
    hb_box_margin_x = int(hb.width * 0.27)
    hb_box_margin_y = int(hb.height * 0.47)
    hb_box_width = int(hb.width * 0.10)
    hb_box_height = int(hb.height * 0.10)

    hb_x1 = hb_x + hb_box_margin_x
    hb_y1 = hb_y + hb_box_margin_y
    hb_x2 = hb_x1 + hb_box_width
    hb_y2 = hb_y1 + hb_box_height

    draw.rectangle([hb_x1, hb_y1, hb_x2, hb_y2], outline=border_color, width=border_width)

    # ---- 2) Purple box around zoom image ----
    z_x1 = zoom_x
    z_y1 = zoom_y
    z_x2 = zoom_x + zm.width
    z_y2 = zoom_y + zm.height

    draw.rectangle([z_x1, z_y1, z_x2, z_y2], outline=border_color, width=border_width)

    # ---- 3) Connector line from hairball box to zoom box ----
    # From middle of right edge of left box to middle of left edge of zoom box
    hb_mid_right = ((hb_x2), (hb_y1 + hb_y2) // 2)
    zoom_mid_left = ((zoom_x), (zoom_y + zm.height // 2))

    draw.line([hb_mid_right, zoom_mid_left], fill=border_color, width=border_width // 2)

    # Save composite
    canvas.save(output_path, quality=95)
    print(f"Saved composite image to {output_path}")


if __name__ == "__main__":
    create_composite()
