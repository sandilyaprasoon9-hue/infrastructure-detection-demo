from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

def generate_architecture_diagram(
        wall_width_cm,
        wall_height_cm,
        pipes,
        filename="architectural_layout.pdf"
    ):
    """
    pipes = [
        {"x_cm": 50, "y_cm": 120, "length_cm": 40},
        {"x_cm": 200, "y_cm": 90, "length_cm": 30}
    ]
    """

    c = canvas.Canvas(filename, pagesize=A4)
    page_w, page_h = A4

    margin = 2*cm

    # Draw wall rectangle
    scale = 0.5   # drawing scale
    wall_w = wall_width_cm * scale
    wall_h = wall_height_cm * scale

    c.setFont("Helvetica", 12)
    c.drawString(margin, page_h - margin, "Architectural Pipe Layout")

    c.rect(margin, margin, wall_w, wall_h)

    # Draw pipes
    for p in pipes:
        x = margin + p["x_cm"] * scale
        y = margin + p["y_cm"] * scale
        length = p["length_cm"] * scale

        c.line(x, y, x + length, y)
        c.drawString(x, y + 5, f'{p["length_cm"]:.1f} cm')

    c.save()
    return filename
