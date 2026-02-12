from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm


def generate_architecture_diagram(wall_w_cm, wall_h_cm, pipes, filename="architecture_layout.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    page_w, page_h = A4

    margin = 2 * cm
    grid = 10   # 1 square = 10 cm
    scale = 0.2  # drawing scale

    # ---------- Draw Graph Grid ----------
    step = grid * scale

    x = margin
    while x < page_w - margin:
        c.setLineWidth(0.2)
        c.line(x, margin, x, page_h - margin)
        x += step

    y = margin
    while y < page_h - margin:
        c.line(margin, y, page_w - margin, y)
        y += step

    # ---------- Draw Wall ----------
    wall_w_draw = wall_w_cm * scale
    wall_h_draw = wall_h_cm * scale

    c.setLineWidth(2)
    c.rect(margin, margin, wall_w_draw, wall_h_draw)

    # ---------- Draw Pipes ----------
    c.setLineWidth(1.5)

    for p in pipes:
        x = margin + p["x_cm"] * scale
        y = margin + p["y_cm"] * scale
        length = p["length_cm"] * scale

        c.line(x, y, x + length, y)
        c.drawString(x, y + 5, f'{p["length_cm"]:.1f} cm')

    c.save()
    return filename
