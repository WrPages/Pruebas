import os
import io
import math
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import discord
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))//idbot
TARGET_CHANNEL_ID = int(os.getenv("TARGET_CHANNEL_ID", "0"))

BASE_DIR = Path(__file__).parent
DETECT_DIR = BASE_DIR / "cards_detect"
HD_DIR = BASE_DIR / "cards_hd"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# Tamaño de referencia de tu imagen de detección.
# En tu ejemplo: 487x427
REFERENCE_W = 487
REFERENCE_H = 427

# Slots de recorte tomados de tu módulo AHK
# formato: (x1, y1, x2, y2)
SLOT_BOXES_REF = [
    (18, 188, 18 + 60, 188 + 80),
    (101, 188, 101 + 60, 188 + 80),
    (183, 188, 183 + 60, 188 + 80),
    (58, 302, 58 + 60, 302 + 80),
    (145, 302, 145 + 60, 302 + 80),
]

# Posiciones de salida para componer la imagen HD
# Se basan en tu AHK, pero aquí puedes ajustar libremente.
CANVAS_W = 2200
CANVAS_H = 2000
CARD_W = 640
CARD_H = 890
DRAW_SLOTS = [
    (120, 50),
    (780, 50),
    (1440, 50),
    (450, 970),
    (1110, 970),
]

# Si quieres que el bot responda sólo cuando vea este texto:
TRIGGER_TEXTS = [
    "God Pack found",
    "[1/5][P][MegaShine]",
]

# -------------------------------------------------------------------
# TEMPLATE INDEX
# -------------------------------------------------------------------

class TemplateCard:
    def __init__(self, name: str, detect_path: Path, hd_path: Path):
        self.name = name
        self.detect_path = detect_path
        self.hd_path = hd_path
        self.detect_bgr = self._load_detect_image(detect_path)
        self.detect_gray = cv2.cvtColor(self.detect_bgr, cv2.COLOR_BGR2GRAY)
        self.detect_hist = self._compute_hist(self.detect_bgr)

    @staticmethod
    def _load_detect_image(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"No se pudo cargar template: {path}")
        return img

    @staticmethod
    def _compute_hist(img_bgr: np.ndarray) -> np.ndarray:
        hist = cv2.calcHist([img_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist


def load_templates() -> List[TemplateCard]:
    templates: List[TemplateCard] = []

    if not DETECT_DIR.exists():
        raise RuntimeError(f"No existe la carpeta: {DETECT_DIR}")
    if not HD_DIR.exists():
        raise RuntimeError(f"No existe la carpeta: {HD_DIR}")

    for detect_file in DETECT_DIR.glob("*.png"):
        name = detect_file.stem
        hd_file = HD_DIR / f"{name}.png"
        if not hd_file.exists():
            print(f"[WARN] No existe HD para {name}, se omite.")
            continue

        try:
            templates.append(TemplateCard(name, detect_file, hd_file))
        except Exception as e:
            print(f"[WARN] Error cargando {name}: {e}")

    if not templates:
        raise RuntimeError("No se cargaron templates válidos.")

    print(f"[INFO] Templates cargados: {len(templates)}")
    return templates


TEMPLATES = load_templates()

# -------------------------------------------------------------------
# IMAGE HELPERS
# -------------------------------------------------------------------

def pil_to_cv(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def scale_box(box: Tuple[int, int, int, int], src_w: int, src_h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    sx = src_w / REFERENCE_W
    sy = src_h / REFERENCE_H
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def crop_slot(img_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))
    return img_bgr[y1:y2, x1:x2].copy()


def compute_hist(img_bgr: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([img_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def compare_images(slot_bgr: np.ndarray, template: TemplateCard) -> float:
    """
    Score menor = mejor.
    Mezcla:
    - diferencia por MSE en gris
    - diferencia de histograma color
    - correlación por template matching
    """
    resized = cv2.resize(slot_bgr, (template.detect_bgr.shape[1], template.detect_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # MSE gris
    diff = gray.astype(np.float32) - template.detect_gray.astype(np.float32)
    mse = float(np.mean(diff ** 2))

    # Hist color
    hist_slot = compute_hist(resized)
    hist_corr = cv2.compareHist(hist_slot.astype(np.float32), template.detect_hist.astype(np.float32), cv2.HISTCMP_CORREL)
    hist_penalty = (1.0 - max(-1.0, min(1.0, hist_corr))) * 1000.0

    # Template matching en gris, mismo tamaño
    res = cv2.matchTemplate(gray, template.detect_gray, cv2.TM_CCOEFF_NORMED)
    match_score = float(res[0][0])
    match_penalty = (1.0 - match_score) * 1000.0

    total_score = (mse * 0.60) + (hist_penalty * 0.20) + (match_penalty * 0.20)
    return total_score


def detect_card(slot_bgr: np.ndarray, templates: List[TemplateCard]) -> Tuple[Optional[TemplateCard], List[Tuple[str, float]]]:
    ranking = []

    for t in templates:
        try:
            score = compare_images(slot_bgr, t)
            ranking.append((t, score))
        except Exception:
            continue

    ranking.sort(key=lambda x: x[1])

    if not ranking:
        return None, []

    best_t, best_score = ranking[0]
    top_debug = [(x[0].name, round(x[1], 3)) for x in ranking[:5]]

    # Validación mínima para evitar falsos positivos absurdos
    # Esto vas a tener que ajustarlo con tus cartas reales.
    if len(ranking) > 1:
        second_score = ranking[1][1]
        gap = second_score - best_score
    else:
        gap = 999999.0

    # Umbrales iniciales razonables
    if best_score < 2500 and gap > 60:
        return best_t, top_debug

    if best_score < 1800:
        return best_t, top_debug

    return None, top_debug


def build_hd_canvas(detected_cards: List[Optional[TemplateCard]]) -> Image.Image:
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (20, 20, 20, 255))

    for i, card in enumerate(detected_cards):
        if card is None:
            continue

        hd = Image.open(card.hd_path).convert("RGBA")
        hd = hd.resize((CARD_W, CARD_H), Image.LANCZOS)

        x, y = DRAW_SLOTS[i]
        canvas.alpha_composite(hd, (x, y))

    return canvas


def extract_slots(source_img: Image.Image) -> List[np.ndarray]:
    img_bgr = pil_to_cv(source_img)
    h, w = img_bgr.shape[:2]

    slots = []
    for ref_box in SLOT_BOXES_REF:
        scaled = scale_box(ref_box, w, h)
        slot = crop_slot(img_bgr, scaled)
        slots.append(slot)
    return slots


def create_debug_contact_sheet(source_img: Image.Image, slots: List[np.ndarray], detected: List[Optional[TemplateCard]]) -> Image.Image:
    thumb_w, thumb_h = 180, 240
    margin = 20
    width = margin + (thumb_w + margin) * 5
    height = 140 + thumb_h + 80
    sheet = Image.new("RGB", (width, height), (28, 28, 28))

    for i, slot in enumerate(slots):
        pil_slot = cv_to_pil(slot).resize((thumb_w, thumb_h), Image.LANCZOS)
        x = margin + i * (thumb_w + margin)
        y = 100
        sheet.paste(pil_slot, (x, y))

        if detected[i]:
            label = detected[i].name
        else:
            label = "No detectada"

        # texto simple sin fuentes custom
        from PIL import ImageDraw
        draw = ImageDraw.Draw(sheet)
        draw.text((x, 20), f"Slot {i+1}", fill=(255, 255, 255))
        draw.text((x, 60), label[:26], fill=(180, 220, 255))

    return sheet


# -------------------------------------------------------------------
# DISCORD BOT
# -------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


def is_target_message(message: discord.Message) -> bool:
    if message.author.bot is False:
        return False

    if CLIENT_ID and message.author.id != CLIENT_ID:
        return False

    if TARGET_CHANNEL_ID and message.channel.id != TARGET_CHANNEL_ID:
        return False

    content = message.content or ""
    return any(t.lower() in content.lower() for t in TRIGGER_TEXTS)


def get_first_image_attachment(message: discord.Message) -> Optional[discord.Attachment]:
    for att in message.attachments:
        ctype = (att.content_type or "").lower()
        if ctype.startswith("image/"):
            return att

        fname = att.filename.lower()
        if fname.endswith((".png", ".jpg", ".jpeg", ".webp")):
            return att
    return None


async def download_image(attachment: discord.Attachment) -> Image.Image:
    data = await attachment.read()
    return Image.open(io.BytesIO(data)).convert("RGBA")


@client.event
async def on_ready():
    print(f"[INFO] Conectado como {client.user}")


@client.event
async def on_message(message: discord.Message):
    try:
        if not is_target_message(message):
            return

        image_attachment = get_first_image_attachment(message)
        if image_attachment is None:
            return

        source_img = await download_image(image_attachment)
        slots = extract_slots(source_img)

        detected_cards: List[Optional[TemplateCard]] = []
        debug_lines = []

        for idx, slot in enumerate(slots):
            card, ranking = detect_card(slot, TEMPLATES)
            detected_cards.append(card)

            if card:
                debug_lines.append(f"Slot {idx+1}: {card.name}")
            else:
                debug_lines.append(f"Slot {idx+1}: no detectada")

            if ranking:
                debug_lines.append(f"  Top: {ranking[:3]}")

        # Si no detectó ninguna, no spamear
        found_count = sum(1 for c in detected_cards if c is not None)
        if found_count == 0:
            return

        hd_canvas = build_hd_canvas(detected_cards)
        debug_sheet = create_debug_contact_sheet(source_img, slots, detected_cards)

        out_main = OUTPUT_DIR / f"gp_hd_{message.id}.png"
        out_debug = OUTPUT_DIR / f"gp_debug_{message.id}.png"
        hd_canvas.save(out_main)
        debug_sheet.save(out_debug)

        files = [
            discord.File(str(out_main), filename="gp_hd.png"),
            discord.File(str(out_debug), filename="gp_debug.png"),
        ]

        summary = "\n".join(debug_lines[:12])
        reply_text = (
            f"Reconstrucción HD del GP\n\n"
            f"Detectadas: {found_count}/5\n"
            f"```{summary[:1800]}```"
        )

        await message.reply(reply_text, files=files, mention_author=False)

    except Exception as e:
        print(f"[ERROR] on_message: {e}")


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("Falta DISCORD_TOKEN en .env")
    client.run(DISCORD_TOKEN)
