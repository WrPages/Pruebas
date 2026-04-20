import os
import io
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import discord
from PIL import Image, ImageDraw

# =========================================================
# VARIABLES DE ENTORNO DESDE RAILWAY
# =========================================================

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
CLIENT_ID = int(os.environ.get("CLIENT_ID", "0"))
TARGET_CHANNEL_ID = int(os.environ.get("TARGET_CHANNEL_ID", "0"))

# =========================================================
# RUTAS
# =========================================================

CARDS_BASE = "/app"
BASE_DIR = Path(CARDS_BASE)
DETECT_DIR = BASE_DIR / "cards_detect"
HD_DIR = BASE_DIR / "cards_hd"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

# Tamaño de referencia de la imagen del grid del GP
# basado en tu ejemplo del screenshot
REFERENCE_W = 487
REFERENCE_H = 427

# Coordenadas de las 5 cartas dentro de la imagen de preview del GP
# formato: (x1, y1, x2, y2)
SLOT_BOXES_REF = [
    (18, 188, 78, 268),    # slot 1
    (101, 188, 161, 268),  # slot 2
    (183, 188, 243, 268),  # slot 3
    (58, 302, 118, 382),   # slot 4
    (145, 302, 205, 382),  # slot 5
]

# Tamaño del canvas de salida
CANVAS_W = 2200
CANVAS_H = 2000

# Tamaño de las cartas HD en la imagen final
CARD_W = 640
CARD_H = 890

# Posiciones donde se dibujan las 5 cartas HD en la salida final
DRAW_SLOTS = [
    (120, 50),
    (780, 50),
    (1440, 50),
    (450, 970),
    (1110, 970),
]

TRIGGER_TEXTS = [
    "God Pack found",
    "[1/5][P][MegaShine]",
]

VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

# =========================================================
# CLASE TEMPLATE
# =========================================================

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
            raise ValueError(f"No se pudo cargar template detect: {path}")
        return img

    @staticmethod
    def _compute_hist(img_bgr: np.ndarray) -> np.ndarray:
        hist = cv2.calcHist([img_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist


# =========================================================
# CARGA DE TEMPLATES
# =========================================================

def load_templates() -> List[TemplateCard]:
    templates: List[TemplateCard] = []

    print(f"[INFO] BASE_DIR: {BASE_DIR}")
    print(f"[INFO] DETECT_DIR existe: {DETECT_DIR.exists()} -> {DETECT_DIR}")
    print(f"[INFO] HD_DIR existe: {HD_DIR.exists()} -> {HD_DIR}")

    if not DETECT_DIR.exists():
        raise RuntimeError(f"No existe la carpeta cards_detect: {DETECT_DIR}")

    if not HD_DIR.exists():
        raise RuntimeError(f"No existe la carpeta cards_hd: {HD_DIR}")

    detect_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        detect_files.extend(DETECT_DIR.glob(ext))

    hd_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        hd_files.extend(HD_DIR.glob(ext))

    print("[INFO] Archivos en cards_detect:")
    for f in detect_files:
        try:
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
        except Exception:
            print(f"  - {f.name} (sin info de tamaño)")

    print("[INFO] Archivos en cards_hd:")
    for f in hd_files:
        try:
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
        except Exception:
            print(f"  - {f.name} (sin info de tamaño)")

    for detect_file in detect_files:
        name = detect_file.stem

        if detect_file.stat().st_size == 0:
            print(f"[WARN] Archivo detect vacío, se omite: {detect_file.name}")
            continue

        possible_hd_files = [
            HD_DIR / f"{name}.png",
            HD_DIR / f"{name}.jpg",
            HD_DIR / f"{name}.jpeg",
            HD_DIR / f"{name}.webp",
        ]

        hd_file = None
        for p in possible_hd_files:
            if p.exists():
                hd_file = p
                break

        if hd_file is None:
            print(f"[WARN] No existe versión HD para {name}")
            continue

        if hd_file.stat().st_size == 0:
            print(f"[WARN] Archivo HD vacío, se omite: {hd_file.name}")
            continue

        try:
            templates.append(TemplateCard(name, detect_file, hd_file))
            print(f"[OK] Template cargado: {name}")
        except Exception as e:
            print(f"[WARN] Error cargando template {name}: {e}")

    print(f"[INFO] Total templates válidos: {len(templates)}")

    if not templates:
        raise RuntimeError("No se cargó ningún template válido en cards_detect/ y cards_hd/")

    return templates


try:
    TEMPLATES = load_templates()
except Exception as e:
    print(f"[ERROR] No se pudieron cargar templates: {e}")
    TEMPLATES = []

# =========================================================
# HELPERS DE IMAGEN
# =========================================================

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

    if x2 <= x1 or y2 <= y1:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    return img_bgr[y1:y2, x1:x2].copy()


def compute_hist(img_bgr: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([img_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def compare_images(slot_bgr: np.ndarray, template: TemplateCard) -> float:
    resized = cv2.resize(
        slot_bgr,
        (template.detect_bgr.shape[1], template.detect_bgr.shape[0]),
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 1) error cuadrático medio en grises
    diff = gray.astype(np.float32) - template.detect_gray.astype(np.float32)
    mse = float(np.mean(diff ** 2))

    # 2) comparación de histograma color
    hist_slot = compute_hist(resized)
    hist_corr = cv2.compareHist(
        hist_slot.astype(np.float32),
        template.detect_hist.astype(np.float32),
        cv2.HISTCMP_CORREL
    )
    hist_penalty = (1.0 - max(-1.0, min(1.0, hist_corr))) * 1000.0

    # 3) template matching
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

    if len(ranking) > 1:
        second_score = ranking[1][1]
        gap = second_score - best_score
    else:
        gap = 999999.0

    # umbrales iniciales
    if best_score < 2500 and gap > 60:
        return best_t, top_debug

    if best_score < 1800:
        return best_t, top_debug

    return None, top_debug


def extract_slots(source_img: Image.Image) -> List[np.ndarray]:
    img_bgr = pil_to_cv(source_img)
    h, w = img_bgr.shape[:2]

    slots = []
    for ref_box in SLOT_BOXES_REF:
        scaled_box = scale_box(ref_box, w, h)
        slot = crop_slot(img_bgr, scaled_box)
        slots.append(slot)

    return slots


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


def create_debug_contact_sheet(source_img: Image.Image, slots: List[np.ndarray], detected_cards: List[Optional[TemplateCard]]) -> Image.Image:
    thumb_w = 180
    thumb_h = 240
    margin = 20

    width = margin + (thumb_w + margin) * 5
    height = 420

    sheet = Image.new("RGB", (width, height), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)

    draw.text((20, 20), "Debug deteccion GP", fill=(255, 255, 255))

    for i, slot in enumerate(slots):
        thumb = cv_to_pil(slot).resize((thumb_w, thumb_h), Image.LANCZOS)
        x = margin + i * (thumb_w + margin)
        y = 100

        sheet.paste(thumb, (x, y))

        draw.text((x, 60), f"Slot {i + 1}", fill=(255, 255, 255))

        label = detected_cards[i].name if detected_cards[i] else "No detectada"
        draw.text((x, 350), label[:24], fill=(180, 220, 255))

    return sheet


def attachment_looks_like_gp_grid(att: discord.Attachment) -> bool:
    filename = att.filename.lower()
    content_type = (att.content_type or "").lower()

    if content_type.startswith("image/"):
        return True

    if filename.endswith(VALID_IMAGE_EXTENSIONS):
        return True

    return False


async def download_pil_image(attachment: discord.Attachment) -> Image.Image:
    data = await attachment.read()
    return Image.open(io.BytesIO(data)).convert("RGBA")


async def get_best_gp_image_attachment(message: discord.Message) -> Optional[discord.Attachment]:
    image_attachments = [att for att in message.attachments if attachment_looks_like_gp_grid(att)]

    if not image_attachments:
        return None

    # Si hay una sola imagen, usar esa
    if len(image_attachments) == 1:
        return image_attachments[0]

    # Si hay varias, elegir la más probable según dimensiones
    best_att = None
    best_score = None

    for att in image_attachments:
        try:
            img = await download_pil_image(att)
            w, h = img.size

            # Penaliza imágenes muy pequeñas o demasiado verticales tipo perfil
            aspect = w / h if h else 1.0

            # Queremos favorecer una imagen más parecida al grid del GP
            # el grid del ejemplo es más ancho que alto o casi cuadrado
            score = abs(w - 487) + abs(h - 427) + (abs(aspect - (487 / 427)) * 100)

            if best_score is None or score < best_score:
                best_score = score
                best_att = att
        except Exception:
            continue

    return best_att if best_att else image_attachments[0]


def is_target_message(message: discord.Message) -> bool:
    if not message.author.bot:
        return False

    if CLIENT_ID and message.author.id != CLIENT_ID:
        return False

    if TARGET_CHANNEL_ID and message.channel.id != TARGET_CHANNEL_ID:
        return False

    content = message.content or ""
    content_lower = content.lower()

    return any(trigger.lower() in content_lower for trigger in TRIGGER_TEXTS)


# =========================================================
# BOT DISCORD
# =========================================================

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"[INFO] Bot conectado como {client.user}")


@client.event
async def on_message(message: discord.Message):
    try:
        if message.author.id == client.user.id:
            return

        if not TEMPLATES:
            print("[WARN] No hay templates cargados.")
            return

        if not is_target_message(message):
            return

        gp_attachment = await get_best_gp_image_attachment(message)
        if gp_attachment is None:
            print("[INFO] Mensaje detectado pero sin imagen válida.")
            return

        source_img = await download_pil_image(gp_attachment)
        slots = extract_slots(source_img)

        detected_cards: List[Optional[TemplateCard]] = []
        debug_lines: List[str] = []

        for idx, slot in enumerate(slots):
            card, ranking = detect_card(slot, TEMPLATES)
            detected_cards.append(card)

            if card:
                debug_lines.append(f"Slot {idx + 1}: {card.name}")
            else:
                debug_lines.append(f"Slot {idx + 1}: no detectada")

            if ranking:
                debug_lines.append(f"Top {idx + 1}: {ranking[:3]}")

        found_count = sum(1 for c in detected_cards if c is not None)

        if found_count == 0:
            print("[INFO] No se detectó ninguna carta.")
            return

        hd_canvas = build_hd_canvas(detected_cards)
        debug_sheet = create_debug_contact_sheet(source_img, slots, detected_cards)

        out_hd = OUTPUT_DIR / f"gp_hd_{message.id}.png"
        out_debug = OUTPUT_DIR / f"gp_debug_{message.id}.png"

        hd_canvas.save(out_hd)
        debug_sheet.save(out_debug)

        reply_text = "Reconstrucción HD del GP\n\n"
        reply_text += f"Detectadas: {found_count}/5\n"
        reply_text += "```" + "\n".join(debug_lines[:20])[:1800] + "```"

        files = [
            discord.File(str(out_hd), filename="gp_hd.png"),
            discord.File(str(out_debug), filename="gp_debug.png"),
        ]

        await message.reply(reply_text, files=files, mention_author=False)

        print(f"[INFO] Procesado mensaje {message.id} - detectadas {found_count}/5")

    except Exception as e:
        print(f"[ERROR] on_message: {e}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("Falta la variable DISCORD_TOKEN en Railway")

    client.run(DISCORD_TOKEN)
