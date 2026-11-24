"""Synthetic label images with known text, one per OCR difficulty. Real artwork + its true text can be added as
bench/real/<name>.(png|jpg|pdf) + <name>.txt (+ optional <name>.langs with codes like "id,en")."""

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
