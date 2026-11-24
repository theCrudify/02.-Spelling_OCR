"""Synthetic label images with known text, one per OCR difficulty. Real artwork + its true text can be added as
bench/real/<name>.(png|jpg|pdf) + <name>.txt (+ optional <name>.langs with codes like "id,en")."""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

F = {
    "sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "cond": "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    "serif": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "ar": "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "th": "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "cjk": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "oblique": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "italic": "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    "script": "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
    "comic": str(Path.home() / ".local/share/fonts/msttcorefonts/Comic_Sans_MS.ttf"),
    "impact": str(Path.home() / ".local/share/fonts/msttcorefonts/Impact.ttf"),
    "light": "/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf",
    "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
}


def font(name: str, size: int):
    return ImageFont.truetype(F[name], size)


def page(lines, size=(1600, 1000), bg="white", fg="black", x=60, y=50, gap=1.6):
    """lines = [(font_name, px, text)] drawn top to bottom."""
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    for name, px, text in lines:
        d.text((x, y), text, fill=fg, font=font(name, px))
        y += int(px * gap)
    return im


ID = "Minuman isotonik untuk menggantikan cairan dan elektrolit tubuh yang hilang."
ID2 = "Simpan di tempat sejuk dan kering, hindarkan dari sinar matahari langsung."
EN = "Isotonic drink to replace body fluids and electrolytes lost through perspiration."
NUTRI = "Energi total 120 kkal, Lemak total 0 g, Protein 0 g, Karbohidrat 30 g, Gula 27,5 g, Natrium 245 mg (16%)"
KOMP = "Komposisi: Air, gula, pengatur keasaman asam sitrat, natrium sitrat, natrium klorida, kalium klorida, kalsium laktat, magnesium karbonat, perisa sintetik sitrus."
AR = "مشروب متساوي التوتر لتعويض سوائل الجسم والأملاح المفقودة"
AR2 = "يحفظ في مكان بارد وجاف بعيدا عن أشعة الشمس المباشرة"
TH = "เครื่องดื่มเกลือแร่สำหรับทดแทนน้ำและเกลือแร่ที่ร่างกายสูญเสียไป"
TH2 = "เก็บในที่แห้งและเย็น หลีกเลี่ยงแสงแดดโดยตรง"
ZH = "等渗饮料 补充身体流失的水分和电解质 请置于阴凉干燥处保存"
JA = "水分と電解質をすばやく補給する飲料です 直射日光を避けて保存してください"
VI = "Nước uống bổ sung ion giúp bù nước và chất điện giải cho cơ thể. Bảo quản nơi khô ráo, thoáng mát."


def rotated():
    im = Image.new("RGB", (1400, 1100), "white")
    texts = {0: "Isi bersih 500 ml horizontal", 180: "Simpan di tempat sejuk", 90: "Kocok sebelum diminum", 270: "Komposisi air gula garam"}
    at = {0: (300, 40), 180: (300, 200), 90: (60, 150), 270: (1250, 150)}
    for a, t in texts.items():
        s = Image.new("RGB", (900, 70), "white")
        ImageDraw.Draw(s).text((10, 10), t, fill="black", font=font("sans", 40))
        im.paste(s.rotate(a, expand=True), at[a])
    return im, " ".join(texts.values())


def colored():
    im = Image.new("RGB", (1600, 760), "white")
    d = ImageDraw.Draw(im)
    rows = [("#0b3d91", "white", "POCARI SWEAT ION SUPPLY DRINK"), ("#f2c200", "#1a1a1a", ID2), ("#d7dbe0", "#8a9099", "Baik digunakan sebelum 12 2027"), ("#c8102e", "#ffe9a8", EN), ("#111111", "#e8e8e8", NUTRI[:60])]
    for i, (bg, fg, t) in enumerate(rows):
        d.rectangle([0, i * 150, 1600, i * 150 + 150], fill=bg)
        d.text((40, i * 150 + 52), t, fill=fg, font=font("sans", 34))
    return im, " ".join(r[2] for r in rows)


def vertical_cjk():
    cols = ["等渗饮料", "补充水分", "阴凉干燥处保存"]
    im = Image.new("RGB", (700, 900), "white")
    d = ImageDraw.Draw(im)
    for c, text in enumerate(cols):
        for r, ch in enumerate(text):
            d.text((520 - c * 150, 60 + r * 100), ch, fill="black", font=font("cjk", 80))
    return im, " ".join(cols)


def photo():
    """Phone photo of a pack: slight skew, blur, sensor noise, JPEG."""
    im = page([("bold", 60, "POCARI SWEAT"), ("sans", 30, ID), ("sans", 30, ID2), ("sans", 26, NUTRI), ("sans", 30, EN)], size=(1700, 520), bg="#f4f1ea")
    im = im.rotate(2.5, expand=True, fillcolor="#f4f1ea").filter(ImageFilter.GaussianBlur(1.1))
    rng = random.Random(7)
    px = im.load()
    for _ in range(40000):
        x, y = rng.randrange(im.width), rng.randrange(im.height)
        v = rng.randrange(-35, 35)
        px[x, y] = tuple(max(0, min(255, c + v)) for c in px[x, y])
    tmp = Path(__file__).parent / "out" / "_photo.jpg"
    tmp.parent.mkdir(exist_ok=True)
    im.save(tmp, quality=55)
    return Image.open(tmp).convert("RGB"), f"POCARI SWEAT {ID} {ID2} {NUTRI} {EN}"


def small_print():
    """A4 at 300 dpi with 5-7 pt text (21-29 px): the usual ingredients / nutrition block."""
    lines = [("bold", 120, "POCARI SWEAT"), ("sans", 29, KOMP), ("sans", 25, NUTRI), ("cond", 21, ID + " " + ID2), ("sans", 21, EN), ("serif", 23, "Diproduksi oleh PT Amerta Indah Otsuka, Sukabumi 43359, Indonesia. BPOM RI MD 265210001234")]
    return page(lines, size=(2480, 3508), y=400, gap=2.4), " ".join(l[2] for l in lines)


def on_path(im, text, name, px, point, fill="black"):
    """Draw text one letter at a time along a path. point(distance) -> (x, y, angle°) of the baseline there."""
    f = font(name, px)
    s = 0.0
    for ch in text:
        w = f.getlength(ch)
        x, y, a = point(s + w / 2)
        tile = Image.new("RGBA", (px * 3, px * 3), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((px * 1.5, px * 2), ch, fill=fill, font=f, anchor="ms")
        tile = tile.rotate(a, center=(px * 1.5, px * 2), resample=Image.BICUBIC)
        im.paste(tile, (int(x - px * 1.5), int(y - px * 2)), tile)
        s += w


def arc(cx, cy, r, text_len, top=True):
    """Text centred on the top of a circle (reads clockwise) or the bottom (reads counter-clockwise), as on a seal."""
    def point(s):
        t = (s - text_len / 2) / r  # radians from the middle of the text
        if top:
            return cx + r * math.sin(t), cy - r * math.cos(t), -math.degrees(t)
        return cx + r * math.sin(t), cy + r * math.cos(t), math.degrees(t)
    return point


def curved():
    """Logo / seal style: gentle arc, tight arc, text around a circle, and a wave."""
    im = Image.new("RGB", (1800, 1500), "white")
    t1, t2, t3, t4, t5 = "POCARI SWEAT ION SUPPLY DRINK", "Minuman isotonik pengganti cairan tubuh", "KUALITAS TERJAMIN", "PRODUK HALAL INDONESIA", "Kocok sebelum diminum sajikan dingin"
    on_path(im, t1, "bold", 60, arc(900, 1900, 1750, font("bold", 60).getlength(t1)))  # gentle arc
    on_path(im, t2, "sans", 44, arc(500, 800, 380, font("sans", 44).getlength(t2)))  # tight arc, top half of a circle
    on_path(im, t3, "bold", 44, arc(1300, 760, 300, font("bold", 44).getlength(t3)))  # seal: top …
    on_path(im, t4, "bold", 40, arc(1300, 760, 340, font("bold", 40).getlength(t4), top=False))  # … and bottom
    on_path(im, t5, "sans", 46, lambda s: (200 + s, 1320 + 45 * math.sin(s / 140), -math.degrees(math.atan(45 / 140 * math.cos(s / 140)))))  # wave
    return im, " ".join([t1, t2, t3, t4, t5])


def slanted():
    """Diagonal text: 8°, 20°, 45° and -30°."""
    im = Image.new("RGB", (1700, 1500), "white")
    rows = [(8, "Isi bersih 500 ml siap minum", (60, 40)), (20, "Simpan di tempat sejuk dan kering", (60, 250)), (45, "Kocok sebelum diminum", (950, 40)), (-30, "Baik digunakan sebelum 12 2027", (60, 800))]
    for a, t, at in rows:
        s = Image.new("RGB", (900, 70), "white")
        ImageDraw.Draw(s).text((10, 10), t, fill="black", font=font("sans", 40))
        im.paste(s.rotate(a, expand=True, fillcolor="white", resample=Image.BICUBIC), at)
    return im, " ".join(r[1] for r in rows)


def typefaces():
    lines = [("oblique", 36, ID), ("italic", 36, ID2), ("script", 46, EN), ("comic", 36, "Rasa jeruk segar, tanpa pengawet, tanpa pewarna buatan."), ("impact", 44, "MINUMAN ISOTONIK RENDAH KALORI 350 ML"), ("light", 36, "Diproduksi oleh PT Amerta Indah Otsuka, Sukabumi"), ("mono", 32, "LOT A1B2C3  EXP 12.2027  BPOM RI MD 265210001234")]
    return page(lines, size=(1900, 620)), " ".join(l[2] for l in lines)


def effects():
    """Outline-only letters, drop shadow on a gradient, text over a busy texture, stretched and squeezed text."""
    im = Image.new("RGB", (1800, 1000), "white")
    d = ImageDraw.Draw(im)
    t1, t2, t3, t4, t5 = "POCARI SWEAT", "Ion Supply Drink", "Simpan di tempat sejuk dan kering", "MINUMAN ISOTONIK", "Kocok sebelum diminum"
    d.text((60, 30), t1, fill="white", font=font("bold", 120), stroke_width=4, stroke_fill="#0b3d91")  # outline only
    for x in range(1800):  # gradient band
        d.line([(x, 200), (x, 380)], fill=(11 + x * 200 // 1800, 61 + x * 150 // 1800, 145))
    d.text((66, 246), t2, fill="#04204f", font=font("bold", 90))
    d.text((60, 240), t2, fill="white", font=font("bold", 90))
    rng = random.Random(3)  # busy background: blobs like fruit / water drops
    for _ in range(260):
        x, y, r = rng.randrange(1800), rng.randrange(400, 600), rng.randrange(8, 40)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(rng.randrange(180, 255), rng.randrange(170, 240), rng.randrange(90, 200)))
    d.text((60, 470), t3, fill="#12224a", font=font("bold", 56))
    for t, y, scale in ((t4, 640, (2.2, 1)), (t5, 800, (0.55, 1))):  # stretched wide / squeezed narrow
        s = Image.new("RGB", (int(font("sans", 60).getlength(t)) + 20, 90), "white")
        ImageDraw.Draw(s).text((10, 10), t, fill="black", font=font("sans", 60))
        im.paste(s.resize((int(s.width * scale[0]), int(s.height * scale[1])), Image.BICUBIC), (60, y))
    return im, " ".join([t1, t2, t3, t4, t5])


def bottle():
    """Photo of a label wrapped round a bottle: text squeezed towards both edges, lines bowed."""
    flat = page([("bold", 70, "POCARI SWEAT"), ("sans", 34, ID), ("sans", 34, ID2), ("sans", 34, EN)], size=(1700, 420), x=120)
    w, h = flat.size
    out = Image.new("RGB", (int(w * 2 / math.pi) + 2, h + 60), "white")
    src = flat.load()
    px = out.load()
    R = w / math.pi
    for X in range(out.width):
        u = (X - R) / R  # -1 … 1 across the visible half of the cylinder
        if abs(u) >= 1:
            continue
        x = int((math.asin(u) + math.pi / 2) * R)
        bow = int(30 * (1 - math.sqrt(1 - u * u)))  # lines sag towards the edges
        for y in range(h):
            if 0 <= x < w:
                px[X, y + bow] = src[x, y]
    return out, f"POCARI SWEAT {ID} {ID2} {EN}"


def cases():
    """name -> (image, expected text, language codes, rotations)"""
    out = {}
    out["latin_clean"] = (page([("bold", 64, "POCARI SWEAT"), ("sans", 34, ID), ("sans", 34, ID2), ("serif", 34, EN)]), f"POCARI SWEAT {ID} {ID2} {EN}", ["id", "en"], False)
    out["large_title"] = (page([("bold", 190, "POCARI SWEAT"), ("sans", 150, "ION SUPPLY"), ("bold", 110, "S O Y J O Y")], size=(1700, 900), gap=1.4), "POCARI SWEAT ION SUPPLY SOYJOY", ["en"], False)
    out["numbers_units"] = (page([("sans", 32, NUTRI), ("sans", 32, "Isi bersih 500 ml  Netto 1,5 L  30 g x 12  pH 3,5  0,5 mg/100 ml"), ("sans", 32, "BPOM RI MD 265210001234  EXP 12.2027  LOT A1B2C3  8992753101207")], size=(1900, 400)), f"{NUTRI} Isi bersih 500 ml Netto 1,5 L 30 g x 12 pH 3,5 0,5 mg/100 ml BPOM RI MD 265210001234 EXP 12.2027 LOT A1B2C3 8992753101207", ["id"], False)
    im, t = small_print()
    out["small_print_a4"] = (im, t, ["id", "en"], False)
    im, t = colored()
    out["colored_backgrounds"] = (im, t, ["id", "en"], False)
    im, t = photo()
    out["photo_degraded"] = (im, t, ["id", "en"], False)
    im, t = rotated()
    out["rotated_4_ways"] = (im, t, ["id"], True)
    for name, make in (("curved_text", curved), ("slanted_text", slanted), ("typefaces", typefaces), ("effects", effects), ("bottle_wrap", bottle)):
        im, t = make()
        out[name] = (im, t, ["id", "en"], name == "slanted_text")
    out["vietnamese"] = (page([("sans", 36, VI)], size=(1900, 200)), VI, ["vi"], False)
    out["mixed_id_vi"] = (page([("sans", 34, ID), ("sans", 34, VI), ("sans", 34, ID2)], size=(1900, 300)), f"{ID} {VI} {ID2}", ["id", "vi"], False)
    out["arabic"] = (page([("ar", 48, AR), ("ar", 48, AR2)], size=(1500, 300)), f"{AR} {AR2}", ["ar"], False)
    out["thai"] = (page([("th", 44, TH), ("th", 44, TH2)], size=(1700, 300)), f"{TH} {TH2}", ["th"], False)
    out["chinese"] = (page([("cjk", 48, ZH)], size=(1700, 200)), ZH, ["zh"], False)
    out["japanese"] = (page([("cjk", 44, JA)], size=(1900, 200)), JA, ["ja"], False)
    im, t = vertical_cjk()
    out["vertical_chinese"] = (im, t, ["zh"], True)
    out["mixed_id_en_ar"] = (page([("bold", 60, "POCARI SWEAT"), ("sans", 34, ID), ("serif", 34, EN), ("ar", 46, AR)], size=(1700, 420)), f"POCARI SWEAT {ID} {EN} {AR}", ["id", "en", "ar"], False)
    return out
