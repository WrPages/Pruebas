import os
import io
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import re

import cv2
import numpy as np
import discord
from PIL import Image, ImageDraw, ImageFont

# =========================================================
# VARIABLES DE ENTORNO DESDE RAILWAY
# =========================================================

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
CLIENT_ID = int(os.environ.get("CLIENT_ID", "0"))
TARGET_CHANNEL_ID = int(os.environ.get("TARGET_CHANNEL_ID", "0"))
FORUM_CHANNEL_ID = int(os.environ.get("FORUM_CHANNEL_ID", "0"))
LOG_CHANNEL_ID = int(os.environ.get("LOG_CHANNEL_ID", "0"))

# =========================================================
# RUTAS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ONE_STAR_DIR = BASE_DIR / "one_star_detect"
TWO_STAR_DIR = BASE_DIR / "two_star_detect"
INVALID_DIR = BASE_DIR / "invalid_detect"
HD_DIR = BASE_DIR / "cards_hd"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("gp_detector")

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

REFERENCE_W = 240
REFERENCE_H = 227
MIN_CONFIDENCE_RATIO = 1.08

# NUEVAS CAJAS DE PRUEBA MÁS GRANDES Y CENTRADAS
# Ajustables después viendo el overlay
SLOT_BOXES_REF = [
    (0, 12 , 76,100),    # slot 1
    (81, 12, 159, 100),  # slot 2
    (163, 12, 242, 100),  # slot 3
    (38, 127, 115, 215),  # slot 4
    (123, 127, 200, 215), # slot 5
]
#SLOT_BOXES_REF = [
 #   (0, 5 , 78, 113),    # slot 1
    #(80, 5, 160, 113),  # slot 2
   # (162, 5, 240, 113),  # slot 3
   # (36, 119, 116, 227),  # slot 4
   # (121, 119, 201, 227), # slot 5
#]

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

TRIGGER_PATTERNS = [
    re.compile(r"god\s*pack", re.IGNORECASE),
    re.compile(r"megashine", re.IGNORECASE),
    re.compile(r"\[1/5\]\[p\]", re.IGNORECASE),
]

VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
PROCESSED_MESSAGES = set()
MAX_SCORE_ACCEPT = 1800
MAX_SCORE_ACCEPT_WITH_GAP = 2500
MIN_SCORE_GAP = 60
SAVE_DEBUG_SLOTS = False
# =========================================================
# CLASE TEMPLATE
# =========================================================

class TemplateCard:
    def __init__(self, name: str, rarity: str, detect_path: Path, hd_path: Path):
        self.name = name
        self.rarity = rarity
        self.detect_path = detect_path
        self.hd_path = hd_path

        self.detect_bgr = self._load_detect_image(detect_path)
        self.detect_gray = cv2.cvtColor(self.detect_bgr, cv2.COLOR_BGR2GRAY)
        self.detect_hist = self._compute_hist(self.detect_bgr)

        self.hd_rgba = Image.open(hd_path).convert("RGBA")
        self.hd_resized = self.hd_rgba.resize((CARD_W, CARD_H), Image.LANCZOS)

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

    detect_groups = [
        ("1★", ONE_STAR_DIR),
        ("2★", TWO_STAR_DIR),
        ("INVALID", INVALID_DIR),
    ]

    logger.info("BASE_DIR: %s", BASE_DIR)
    logger.info("HD_DIR existe: %s -> %s", HD_DIR.exists(), HD_DIR)

    if not HD_DIR.exists():
        raise RuntimeError(f"No existe la carpeta cards_hd: {HD_DIR}")

    valid_suffixes = {".png", ".jpg", ".jpeg", ".webp"}

    hd_files = sorted(
        [
            p for p in HD_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in valid_suffixes
        ],
        key=lambda p: p.name.lower()
    )

    for rarity, detect_dir in detect_groups:
        logger.info("Leyendo templates %s desde %s", rarity, detect_dir)

        if not detect_dir.exists():
            logger.warning("No existe carpeta de detección %s: %s", rarity, detect_dir)
            continue

        detect_files = sorted(
            [
                p for p in detect_dir.iterdir()
                if p.is_file() and p.suffix.lower() in valid_suffixes
            ],
            key=lambda p: p.name.lower()
        )

        if not detect_files:
            logger.warning("Carpeta vacía para %s: %s", rarity, detect_dir)
            continue

        for detect_file in detect_files:
            name = detect_file.stem

            if detect_file.stat().st_size == 0:
                logger.warning("Archivo detect vacío, se omite: %s", detect_file.name)
                continue

            possible_hd_files = [
                p for p in hd_files
                if p.stem.lower() == name.lower()
            ]

            if not possible_hd_files:
                logger.warning("No existe versión HD para %s", name)
                continue

            hd_file = possible_hd_files[0]

            if hd_file.stat().st_size == 0:
                logger.warning("Archivo HD vacío, se omite: %s", hd_file.name)
                continue

            try:
                templates.append(TemplateCard(name, rarity, detect_file, hd_file))
                logger.info("Template cargado: %s | rareza=%s", name, rarity)
            except Exception as e:
                logger.warning("Error cargando template %s (%s): %s", name, rarity, e)

    logger.info("Total templates válidos: %s", len(templates))

    if not templates:
        raise RuntimeError("No se cargó ningún template válido en las carpetas de detección")

    return templates


try:
    TEMPLATES = load_templates()
except Exception as e:
    logger.exception("No se pudieron cargar templates: %s", e)
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
    
def preprocess_slot(slot_bgr: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(slot_bgr, (3, 3), 0)
    yuv = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def compare_images(slot_bgr: np.ndarray, template: TemplateCard) -> float:
    resized = cv2.resize(
        slot_bgr,
        (template.detect_bgr.shape[1], template.detect_bgr.shape[0]),
        interpolation=cv2.INTER_AREA
    )
    resized = preprocess_slot(resized)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)
    tpl_gray = cv2.equalizeHist(template.detect_gray)

    diff = gray.astype(np.float32) - tpl_gray.astype(np.float32)
    mse = float(np.mean(diff ** 2))

    hist_slot = compute_hist(resized)
    hist_corr = cv2.compareHist(
        hist_slot.astype(np.float32),
        template.detect_hist.astype(np.float32),
        cv2.HISTCMP_CORREL
    )
    hist_penalty = (1.0 - max(-1.0, min(1.0, hist_corr))) * 1000.0

    l2_distance = cv2.norm(gray, tpl_gray, cv2.NORM_L2) / gray.size

    total_score = (mse * 0.50) + (hist_penalty * 0.20) + (l2_distance * 1000.0 * 0.30)
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
    top_debug = [
        (f"{x[0].name} [{x[0].rarity}]", round(x[1], 3))
        for x in ranking[:5]
    ]

    if len(ranking) > 1:
        second_score = ranking[1][1]
        gap = second_score - best_score
        ratio = second_score / max(best_score, 1e-6)
    else:
        gap = 999999.0
        ratio = 999999.0

    if best_score < MAX_SCORE_ACCEPT:
        return best_t, top_debug

    if best_score < MAX_SCORE_ACCEPT_WITH_GAP and gap > MIN_SCORE_GAP and ratio > MIN_CONFIDENCE_RATIO:
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
    #canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (20, 20, 20, 255))
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))

    # ===== GRID CONFIG =====
    cols = 3
    rows = 2
    gap_x = 40
    gap_y = 60

    # calcular tamaño total del grid
    total_w = cols * CARD_W + (cols - 1) * gap_x
    total_h = rows * CARD_H + (rows - 1) * gap_y

    # 👇 CENTRADO AUTOMÁTICO
    start_x = (CANVAS_W - total_w) // 2
    start_y = (CANVAS_H - total_h) // 2

    positions = [
        (0, 0), (1, 0), (2, 0),
        (0.5, 1), (1.5, 1)  # fila de abajo centrada
    ]

    for i, card in enumerate(detected_cards):
        if card is None:
            continue

        col, row = positions[i]

        x = int(start_x + col * (CARD_W + gap_x))
        y = int(start_y + row * (CARD_H + gap_y))

        canvas.alpha_composite(card.hd_resized, (x, y))

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

        label = f"{detected_cards[i].name} [{detected_cards[i].rarity}]" if detected_cards[i] else "No detectada"
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

async def get_best_gp_image_attachment(message: discord.Message) -> Optional[Tuple[discord.Attachment, Image.Image]]:
    image_attachments = [att for att in message.attachments if attachment_looks_like_gp_grid(att)]

    if not image_attachments:
        return None

    preferred_keywords = ["screenshot", "screen", "pack", "packs", "godpack", "gp"]

    best_att = None
    best_img = None
    best_score = None

    for att in image_attachments:
        name = att.filename.lower()
        if any(k in name for k in preferred_keywords):
            try:
                img = await download_pil_image(att)
                logger.info("Attachment elegido por nombre: %s", att.filename)
                return att, img
            except Exception as e:
                logger.warning("Error cargando attachment %s: %s", att.filename, e)

    for att in image_attachments:
        try:
            img = await download_pil_image(att)
            w, h = img.size
            aspect = w / h if h else 1.0

            penalty = 0
            if aspect < 0.7:
                penalty += 1000

            score = abs(w - 240) + abs(h - 227) + abs(aspect - (240 / 227)) * 100 + penalty

            logger.info("Attachment candidato: %s size=%sx%s score=%s", att.filename, w, h, score)

            if best_score is None or score < best_score:
                best_score = score
                best_att = att
                best_img = img

        except Exception as e:
            logger.warning("Error analizando attachment %s: %s", att.filename, e)

    if best_att is not None and best_img is not None:
        logger.info("Attachment elegido por tamaño: %s", best_att.filename)
        return best_att, best_img

    return None



def is_target_message(message: discord.Message) -> bool:
    if message.webhook_id is None:
        return False

    if TARGET_CHANNEL_ID and message.channel.id != TARGET_CHANNEL_ID:
        return False

    content = message.content or ""

    if any(pattern.search(content) for pattern in TRIGGER_PATTERNS):
        return True

    return len(message.attachments) > 0

def parse_heartbeat_metadata(content: str) -> dict:
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    result = {
        "obtainer_user": None,
        "bot_name": None,
        "game_id": None,
        "packs_count": None,
        "filename": None,
        "raw_pack_line": None,
    }

    for line in lines:
        # 👇 ESTE ES EL USUARIO REAL
        m = re.match(r"^(.+?)\s*\((\d+)\)$", line)
        if m:
            result["obtainer_user"] = m.group(1).strip()
            result["bot_name"] = m.group(1).strip()
            result["game_id"] = m.group(2).strip()
            continue

        m = re.search(r"(\[\d/5\]\[(\d+)P\]\[MegaShine\])", line, re.IGNORECASE)
        if m:
            result["raw_pack_line"] = m.group(1)
            result["packs_count"] = int(m.group(2))
            continue

        m = re.match(r"^File name:\s*(.+)$", line, re.IGNORECASE)
        if m:
            result["filename"] = m.group(1).strip()

    return result



def build_pack_rarity_label(detected_cards: List[Optional["TemplateCard"]]) -> str:
    one_star = 0
    two_star = 0
    invalid = 0

    for card in detected_cards:
        if card is None:
            continue
        if card.rarity == "1★":
            one_star += 1
        elif card.rarity == "2★":
            two_star += 1
        else:
            invalid += 1

    if invalid > 0:
        return f"[INVALID:{invalid}/5]"

    return f"[{two_star}/5]"

def get_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def build_final_poster(
    hd_canvas: Image.Image,
    pack_label: str,
    packs_count: Optional[int],
    bot_name: Optional[str],
) -> Image.Image:
    footer_h = 110
    final_img = Image.new("RGBA", (hd_canvas.width, hd_canvas.height + footer_h), (20, 20, 20, 255))
    final_img.alpha_composite(hd_canvas, (0, 0))

    draw = ImageDraw.Draw(final_img)
    font = get_font(72)

    packs_text = f"[{packs_count}P]" if packs_count is not None else "[?P]"
    bot_text = bot_name or "UnknownBot"
    footer_text = f"{pack_label}   {packs_text}   {bot_text}"

    bbox = draw.textbbox((0, 0), footer_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (final_img.width - text_w) // 2
    y = hd_canvas.height + (footer_h - text_h) // 2

    draw.text((x, y), footer_text, fill=(235, 235, 235), font=font)
    return final_img

def build_forum_post_text(meta: dict, pack_label: str) -> str:
    obtainer = f"@{meta['obtainer_user']}" if meta.get("obtainer_user") else "@desconocido"
    bot_name = meta.get("bot_name") or "UnknownBot"
    game_id = meta.get("game_id") or "UnknownID"
    packs_count = meta.get("packs_count")
    packs_text = f"[{packs_count}P]" if packs_count is not None else "[?P]"
    filename = meta.get("filename") or "unknown_file.xml"

    return f"""```
{obtainer}
{bot_name} ({game_id})
{pack_label}{packs_text}[MegaShine]
{filename}
```"""

def build_post_title(meta: dict, pack_label: str) -> str:
    packs_count = meta.get("packs_count")
    packs_text = f"[{packs_count}P]" if packs_count is not None else "[?P]"
    bot_name = meta.get("bot_name") or "UnknownBot"
    return f"{pack_label} {packs_text} {bot_name}"

async def create_forum_post_with_image(
    client: discord.Client,
    title: str,
    body_text: str,
    image_path: Path,
) -> Optional[str]:
    if not FORUM_CHANNEL_ID:
        logger.warning("FORUM_CHANNEL_ID no configurado")
        return None

    channel = client.get_channel(FORUM_CHANNEL_ID)
    if channel is None:
        try:
            channel = await client.fetch_channel(FORUM_CHANNEL_ID)
        except Exception as e:
            logger.exception("No se pudo obtener el canal foro: %s", e)
            return None

    if not isinstance(channel, discord.ForumChannel):
        logger.error("FORUM_CHANNEL_ID no corresponde a un ForumChannel")
        return None

    try:
        file = discord.File(str(image_path), filename=image_path.name)

        created = await channel.create_thread(
            name=title,
            content=body_text,
            file=file,
        )

        thread = created.thread if hasattr(created, "thread") else created
        return thread.jump_url

    except Exception as e:
        logger.exception("No se pudo crear el post del foro: %s", e)
        return None

class ForumLinkView(discord.ui.View):
    def __init__(self, post_url: str, meta: dict, pack_label: str):
        super().__init__(timeout=None)

        packs = meta.get("packs_count", "?")
        bot = meta.get("bot_name", "Bot")

        label = f"{pack_label} [{packs}P] {bot}"

        self.add_item(
            discord.ui.Button(
                label=label[:80],
                style=discord.ButtonStyle.link,
                url=post_url
            )
        )


def build_log_summary(meta: dict, pack_label: str, debug_lines: List[str]) -> str:
    obtainer = f"@{meta['obtainer_user']}" if meta.get("obtainer_user") else "@desconocido"
    bot_name = meta.get("bot_name") or "UnknownBot"
    game_id = meta.get("game_id") or "UnknownID"
    packs_count = meta.get("packs_count")
    packs_text = f"[{packs_count}P]" if packs_count is not None else "[?P]"
    filename = meta.get("filename") or "unknown_file.xml"

    compact_top = []
    for line in debug_lines:
        if line.startswith("Slot ") or line.startswith("Top "):
            compact_top.append(line)

    compact_top = compact_top[:10]

    return (
        f"**Resumen GP**\n"
        f"```"
        f"{obtainer}\n"
        f"{bot_name} ({game_id})\n"
        f"{pack_label}{packs_text}[MegaShine]\n"
        f"{filename}\n\n"
        + "\n".join(compact_top) +
        f"```"
    )


async def delete_message_later(message: discord.Message, delay_seconds: int = 172800):
    try:
        await asyncio.sleep(delay_seconds)
        await message.delete()
    except Exception:
        pass
        
def process_gp_image(source_img: Image.Image, message_id: int, heartbeat_text: str) -> dict:
    logger.info("Procesando imagen para message_id=%s", message_id)

    debug_source = OUTPUT_DIR / f"debug_source_{message_id}.png"
    source_img.save(debug_source)

    box_overlay = create_box_overlay(source_img)
    overlay_path = OUTPUT_DIR / f"box_overlay_{message_id}.png"
    box_overlay.save(overlay_path)

    slots = extract_slots(source_img)

    if SAVE_DEBUG_SLOTS:
        for i, slot in enumerate(slots):
            slot_path = OUTPUT_DIR / f"debug_slot_{message_id}_{i+1}.png"
            cv_to_pil(slot).save(slot_path)
            logger.debug("Guardado slot %s: %s", i + 1, slot_path)

    detected_cards: List[Optional[TemplateCard]] = []
    debug_lines: List[str] = []

    for idx, slot in enumerate(slots):
        card, ranking = detect_card(slot, TEMPLATES)
        logger.info("Slot %s ranking: %s", idx + 1, ranking[:5])

        detected_cards.append(card)

        if card:
            debug_lines.append(f"Slot {idx + 1}: {card.name} | rareza={card.rarity}")
        else:
            debug_lines.append(f"Slot {idx + 1}: no detectada")

        if ranking:
            top_text = [f"{name} ({score})" for name, score in ranking[:3]]
            debug_lines.append(f"Top {idx + 1}: {top_text}")

    found_count = sum(1 for c in detected_cards if c is not None)
    logger.info("Detectadas: %s/5", found_count)

    meta = parse_heartbeat_metadata(heartbeat_text)
    pack_label = build_pack_rarity_label(detected_cards)

    debug_sheet = create_debug_contact_sheet(source_img, slots, detected_cards)
    out_debug = OUTPUT_DIR / f"gp_debug_{message_id}.png"
    debug_sheet.save(out_debug)

    if found_count == 0:
        return {
            "found_count": found_count,
            "overlay_path": overlay_path,
            "debug_path": out_debug,
            "reply_text": "No se detectó ninguna carta",
            "debug_lines": debug_lines,
            "files": [
                discord.File(str(overlay_path), filename="box_overlay.png"),
                discord.File(str(out_debug), filename="gp_debug.png"),
            ],
            "pack_label": pack_label,
            "heartbeat_meta": meta,
            "final_image_path": None,
        }

    hd_canvas = build_hd_canvas(detected_cards)

    out_hd = OUTPUT_DIR / f"gp_hd_{message_id}.png"
    hd_canvas.save(out_hd)

    reply_text = "Reconstrucción HD del GP\n\n"
    reply_text += f"{pack_label} "
    reply_text += f"[{meta.get('packs_count', '?')}P] "
    reply_text += f"{meta.get('bot_name', 'UnknownBot')}\n"
    reply_text += "```" + "\n".join(debug_lines[:20])[:1800] + "```"

    return {
        "found_count": found_count,
        "overlay_path": overlay_path,
        "debug_path": out_debug,
        "reply_text": reply_text,
        "debug_lines": debug_lines,
        "files": [
            discord.File(str(out_hd), filename="gp_hd.png"),
            discord.File(str(overlay_path), filename="box_overlay.png"),
            discord.File(str(out_debug), filename="gp_debug.png"),
        ],
        "pack_label": pack_label,
        "heartbeat_meta": meta,
        "final_image_path": out_hd,
    }


# =========================================================
# BOT DISCORD
# =========================================================

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    logger.info("Bot conectado como %s", client.user)


@client.event
async def on_message(message: discord.Message):
    try:
        logger.info(
            "on_message author=%s author_id=%s bot=%s webhook_id=%s channel_id=%s content=%r attachments=%s",
            message.author,
            message.author.id,
            message.author.bot,
            message.webhook_id,
            message.channel.id,
            message.content,
            [a.filename for a in message.attachments]
        )

        if message.author.id == client.user.id:
            logger.info("Ignorado: mensaje del propio bot")
            return

        if message.id in PROCESSED_MESSAGES:
            logger.info("Ignorado: mensaje ya procesado %s", message.id)
            return

        if not TEMPLATES:
            logger.info("Ignorado: no hay templates cargados")
            return

        if not is_target_message(message):
            logger.info("Ignorado: no coincide con filtro webhook/canal/trigger")
            return

        logger.info("Mensaje webhook objetivo detectado, procesando...")

        gp_result = await get_best_gp_image_attachment(message)
        if gp_result is None:
            logger.info("No se encontró imagen válida en attachments")
            return

        gp_attachment, source_img = gp_result
        logger.info("gp_attachment seleccionado: %s", gp_attachment.filename)
        logger.info("source_img size: %s", source_img.size)

        PROCESSED_MESSAGES.add(message.id)

        if len(PROCESSED_MESSAGES) > 1000:
            PROCESSED_MESSAGES.clear()

        result = await asyncio.to_thread(process_gp_image, source_img, message.id, message.content)

        post_url = None
        if result["final_image_path"] is not None:
            post_title = build_post_title(result["heartbeat_meta"], result["pack_label"])
            post_body = build_forum_post_text(result["heartbeat_meta"], result["pack_label"])

            post_url = await create_forum_post_with_image(
                client=client,
                title=post_title,
                body_text=post_body,
                image_path=result["final_image_path"],
            )

        view = ForumLinkView(
            post_url,
            result["heartbeat_meta"],
            result["pack_label"]
        ) if post_url else None

        # =========================
        # 1. RESPUESTA LIMPIA EN CANAL ORIGINAL
        # =========================
        original_files = []
        if result["final_image_path"] is not None:
            original_files.append(
                discord.File(str(result["final_image_path"]), filename="gp_hd.png")
            )

        await message.reply(
            files=original_files,
            view=view,
            mention_author=False
        )

        # =========================
        # 2. ENVÍO COMPLETO A CANAL DE REGISTRO
        # =========================
        if LOG_CHANNEL_ID:
            log_channel = client.get_channel(LOG_CHANNEL_ID)
            if log_channel is None:
                try:
                    log_channel = await client.fetch_channel(LOG_CHANNEL_ID)
                except Exception as e:
                    logger.exception("No se pudo obtener el canal log: %s", e)
                    log_channel = None

            if log_channel is not None:
                log_summary = build_log_summary(
                    result["heartbeat_meta"],
                    result["pack_label"],
                    result.get("debug_lines", [])
                )

                log_files = [
                    discord.File(str(result["overlay_path"]), filename="box_overlay.png"),
                    discord.File(str(result["debug_path"]), filename="gp_debug.png"),
                ]

                sent_log = await log_channel.send(
                    content=(
                        f"{log_summary}\n"
                        f"**Mensaje original:**\n"
                        f"```{message.content[:1800]}```"
                    ),
                    files=log_files
                )

                asyncio.create_task(delete_message_later(sent_log, 172800))

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("Falta la variable DISCORD_TOKEN en Railway")

    client.run(DISCORD_TOKEN)
