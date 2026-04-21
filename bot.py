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

BASE_DIR = Path(__file__).resolve().parent
DETECT_DIR = BASE_DIR / "cards_detect"
HD_DIR = BASE_DIR / "cards_hd"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

REFERENCE_W = 240
REFERENCE_H = 227

# NUEVAS CAJAS DE PRUEBA MÁS GRANDES Y CENTRADAS
# Ajustables después viendo el overlay
SLOT_BOXES_REF = [
    (0, 5 , 78, 113),    # slot 1
    (80, 5, 160, 113),  # slot 2
    (162, 5, 240, 113),  # slot 3
    (36, 119, 116, 227),  # slot 4
    (121, 129, 201, 217), # slot 5
]

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
        if img is not None:
            return img

        try:
            pil_img = Image.open(path).convert("RGB")
            rgb = np.array(pil_img)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr
        except Exception as e:
            raise ValueError(f"No se pudo cargar template detect: {path} | detalle: {e}")

    @staticmethod
    def _compute_hist(img_bgr: np.ndarray) -> np.ndarray:
        hist = cv2.calcHist(
            [img_bgr],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 256, 0, 256, 0, 256]
        )
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

    valid_suffixes = {".png", ".jpg", ".jpeg", ".webp"}

    detect_files = sorted(
        [
            p for p in DETECT_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in valid_suffixes
        ],
        key=lambda p: p.name.lower()
    )

    hd_files = sorted(
        [
            p for p in HD_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in valid_suffixes
        ],
        key=lambda p: p.name.lower()
    )

    print("[INFO] Archivos en cards_detect:")
    if not detect_files:
        print("  - (vacío)")
    for f in detect_files:
        try:
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
        except Exception:
            print(f"  - {f.name} (sin info de tamaño)")

    print("[INFO] Archivos en cards_hd:")
    if not hd_files:
        print("  - (vacío)")
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
            p for p in hd_files
            if p.stem.lower() == name.lower()
        ]

        if not possible_hd_files:
            print(f"[WARN] No existe versión HD para {name}")
            continue

        hd_file = possible_hd_files[0]

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
    hist = cv2.calcHist(
        [img_bgr],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def compare_images(slot_bgr: np.ndarray, template: TemplateCard) -> float:
    resized = cv2.resize(
        slot_bgr,
        (template.detect_bgr.shape[1], template.detect_bgr.shape[0]),
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    diff = gray.astype(np.float32) - template.detect_gray.astype(np.float32)
    mse = float(np.mean(diff ** 2))

    hist_slot = compute_hist(resized)
    hist_corr = cv2.compareHist(
        hist_slot.astype(np.float32),
        template.detect_hist.astype(np.float32),
        cv2.HISTCMP_CORREL
    )
    hist_penalty = (1.0 - max(-1.0, min(1.0, hist_corr))) * 1000.0

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


def create_debug_contact_sheet(
    source_img: Image.Image,
    slots: List[np.ndarray],
    detected_cards: List[Optional[TemplateCard]]
) -> Image.Image:
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


def create_box_overlay(source_img: Image.Image) -> Image.Image:
    overlay = source_img.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)

    w, h = overlay.size

    for i, ref_box in enumerate(SLOT_BOXES_REF):
        x1, y1, x2, y2 = scale_box(ref_box, w, h)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=4)
        draw.text((x1 + 5, y1 + 5), f"S{i+1}", fill=(255, 255, 0))

    return overlay


def attachment_looks_like_gp_grid(att: discord.Attachment) -> bool:
    filename = att.filename.lower()
    content_type = (att.content_type or "").lower()

    if content_type.startswith("image/"):
        return True

    if filename.endswith(tuple(ext.lower() for ext in VALID_IMAGE_EXTENSIONS)):
        return True

    return False


async def download_pil_image(attachment: discord.Attachment) -> Image.Image:
    data = await attachment.read()
    return Image.open(io.BytesIO(data)).convert("RGBA")


async def get_best_gp_image_attachment(message: discord.Message) -> Optional[discord.Attachment]:
    image_attachments = [att for att in message.attachments if attachment_looks_like_gp_grid(att)]

    if not image_attachments:
        return None

    preferred_keywords = ["screenshot", "screen", "pack", "packs", "godpack", "gp"]

    for att in image_attachments:
        name = att.filename.lower()
        if any(k in name for k in preferred_keywords):
            print(f"[DEBUG] Attachment elegido por nombre: {att.filename}")
            return att

    best_att = None
    best_score = None

    for att in image_attachments:
        try:
            img = await download_pil_image(att)
            w, h = img.size
            aspect = w / h if h else 1.0

            penalty = 0
            if aspect < 0.7:
                penalty += 1000

            score = abs(w - 240) + abs(h - 227) + abs(aspect - (240 / 227)) * 100 + penalty

            print(f"[DEBUG] Attachment candidato: {att.filename} size={w}x{h} score={score}")

            if best_score is None or score < best_score:
                best_score = score
                best_att = att
        except Exception as e:
            print(f"[WARN] Error analizando attachment {att.filename}: {e}")

    if best_att:
        print(f"[DEBUG] Attachment elegido por tamaño: {best_att.filename}")

    return best_att if best_att else image_attachments[0]


def is_target_message(message: discord.Message) -> bool:
    if message.webhook_id is None:
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
        print(
            f"[DEBUG] on_message author={message.author} "
            f"author_id={message.author.id} "
            f"bot={message.author.bot} "
            f"webhook_id={message.webhook_id} "
            f"channel_id={message.channel.id} "
            f"content={repr(message.content)} "
            f"attachments={[a.filename for a in message.attachments]}"
        )

        if message.author.id == client.user.id:
            print("[DEBUG] Ignorado: mensaje del propio bot")
            return

        if not TEMPLATES:
            print("[DEBUG] Ignorado: no hay templates cargados")
            return

        if not is_target_message(message):
            print("[DEBUG] Ignorado: no coincide con filtro webhook/canal/trigger")
            return

        print("[DEBUG] Mensaje webhook objetivo detectado, procesando...")

        gp_attachment = await get_best_gp_image_attachment(message)
        if gp_attachment is None:
            print("[DEBUG] No se encontró imagen válida en attachments")
            return

        print(f"[DEBUG] gp_attachment seleccionado: {gp_attachment.filename}")

        source_img = await download_pil_image(gp_attachment)
        print(f"[DEBUG] source_img size: {source_img.size}")

        debug_source = OUTPUT_DIR / f"debug_source_{message.id}.png"
        source_img.save(debug_source)

        box_overlay = create_box_overlay(source_img)
        overlay_path = OUTPUT_DIR / f"box_overlay_{message.id}.png"
        box_overlay.save(overlay_path)

        slots = extract_slots(source_img)

        for i, slot in enumerate(slots):
            slot_path = OUTPUT_DIR / f"debug_slot_{message.id}_{i+1}.png"
            cv_to_pil(slot).save(slot_path)
            print(f"[DEBUG] Guardado slot {i+1}: {slot_path}")

        detected_cards: List[Optional[TemplateCard]] = []
        debug_lines: List[str] = []

        for idx, slot in enumerate(slots):
            card, ranking = detect_card(slot, TEMPLATES)
            print(f"[DEBUG] Slot {idx+1} ranking: {ranking[:5]}")

            detected_cards.append(card)

            if card:
                debug_lines.append(f"Slot {idx + 1}: {card.name}")
            else:
                debug_lines.append(f"Slot {idx + 1}: no detectada")

            if ranking:
                debug_lines.append(f"Top {idx + 1}: {ranking[:3]}")

        found_count = sum(1 for c in detected_cards if c is not None)
        print(f"[DEBUG] Detectadas: {found_count}/5")

        debug_sheet = create_debug_contact_sheet(source_img, slots, detected_cards)
        out_debug = OUTPUT_DIR / f"gp_debug_{message.id}.png"
        debug_sheet.save(out_debug)

        if found_count == 0:
            print("[INFO] No se detectó ninguna carta.")

            await message.reply(
                "No se detectó ninguna carta. Revisa overlay y debug.",
                files=[
                    discord.File(str(overlay_path), filename="box_overlay.png"),
                    discord.File(str(out_debug), filename="gp_debug.png"),
                ],
                mention_author=False
            )
            return

        hd_canvas = build_hd_canvas(detected_cards)
        out_hd = OUTPUT_DIR / f"gp_hd_{message.id}.png"
        hd_canvas.save(out_hd)

        reply_text = "Reconstrucción HD del GP\n\n"
        reply_text += f"Detectadas: {found_count}/5\n"
        reply_text += "```" + "\n".join(debug_lines[:20])[:1800] + "```"

        files = [
            discord.File(str(out_hd), filename="gp_hd.png"),
            discord.File(str(overlay_path), filename="box_overlay.png"),
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
