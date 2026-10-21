"""
Copyright (c) 2026 Badan Riset dan Inovasi Nasional (BRIN)
THIS SOFTWARE SOURCE CODE AND ANY EXECUTABLE DERIVED THEREOF ARE PROPRIETARY
TO BRIN, AS APPLICABLE, AND SHALL NOT BE USED IN ANY WAY
OTHER THAN BEFOREHAND AGREED ON BY BRIN, NOR BE REPRODUCED
OR DISCLOSED TO THIRD PARTIES WITHOUT PRIOR WRITTEN AUTHORIZATION BY
BRIN, AS APPLICABLE.

Author      : Chandra
Version     : 0.1.0
Created     : 21-10-2026
Description : Modul logging berwarna (ANSI) untuk terminal dan plain text untuk file log.
              Bebas emoticon/emoji, mendukung formatting reasoning per-sampel terstruktur.
"""

import logging
import os
import re
import sys
import typing as t
from datetime import datetime


class AnsiColor:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


ANSI_REGEX = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(text: str) -> str:
    """Menghapus kode escape ANSI agar file log tetap teks polos yang bersih."""
    return ANSI_REGEX.sub("", text)


class PlainFileFormatter(logging.Formatter):
    """Formatter untuk berkas file log yang secara otomatis membersihkan kode ANSI."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return strip_ansi(original)


class ColoredConsoleFormatter(logging.Formatter):
    """Formatter terminal berwarna dengan timestamp dan level badge yang rapi."""

    LEVEL_COLORS = {
        logging.DEBUG: AnsiColor.DIM + AnsiColor.WHITE,
        logging.INFO: AnsiColor.BRIGHT_CYAN,
        logging.WARNING: AnsiColor.BRIGHT_YELLOW,
        logging.ERROR: AnsiColor.BRIGHT_RED,
        logging.CRITICAL: AnsiColor.BOLD + AnsiColor.BRIGHT_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, AnsiColor.WHITE)
        time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level_name = f"[{record.levelname}]"
        
        # Format: [11:15:30] [INFO] message
        header = f"{AnsiColor.DIM}[{time_str}]{AnsiColor.RESET} {color}{level_name}{AnsiColor.RESET} "
        return f"{header}{record.getMessage()}"


def colorize_score(score: t.Any, width: int = 6) -> str:
    """Mewarnai angka skor berdasarkan ambang batas performa."""
    if score is None or score == "":
        return f"{AnsiColor.DIM}N/A{AnsiColor.RESET}"
    if isinstance(score, str) and score.startswith("ERROR"):
        return f"{AnsiColor.BRIGHT_RED}{score}{AnsiColor.RESET}"
    try:
        val = float(score)
        val_str = f"{val:.4f}"
        if val >= 0.70:
            return f"{AnsiColor.BOLD}{AnsiColor.BRIGHT_GREEN}{val_str}{AnsiColor.RESET}"
        elif val >= 0.40:
            return f"{AnsiColor.BOLD}{AnsiColor.BRIGHT_YELLOW}{val_str}{AnsiColor.RESET}"
        else:
            return f"{AnsiColor.BOLD}{AnsiColor.BRIGHT_RED}{val_str}{AnsiColor.RESET}"
    except (ValueError, TypeError):
        return str(score)


def format_reasoning_items(reasoning_text: str, indent: str = "    ") -> t.List[str]:
    """
    Memformat baris reasoning menjadi daftar baris:
    - [OK]   |  Pernyataan ... | Alasan: ...
    dengan pewarnaan tag dan kolom pipe '|' yang teratur rapi.
    """
    if not reasoning_text or reasoning_text.strip() == "":
        return [f"{indent}{AnsiColor.DIM}(tidak ada rincian reasoning){AnsiColor.RESET}"]

    raw_items = [item.strip() for item in reasoning_text.split(" | ") if item.strip()]
    formatted = []
    c = AnsiColor
    pipe = f"{c.CYAN}|{c.RESET}"

    for item in raw_items:
        m = re.match(r"^\[(OK|FAIL|TP|FP|FN)\]\s*(.*)", item, re.DOTALL)
        if m:
            tag = m.group(1)
            body = m.group(2).strip()

            if tag == "OK":
                tag_str = f"{c.BOLD}{c.BRIGHT_GREEN}[OK]{c.RESET}  "
            elif tag == "FAIL":
                tag_str = f"{c.BOLD}{c.BRIGHT_RED}[FAIL]{c.RESET}"
            elif tag == "TP":
                tag_str = f"{c.BOLD}{c.BRIGHT_GREEN}[TP]{c.RESET}  "
            elif tag == "FP":
                tag_str = f"{c.BOLD}{c.BRIGHT_RED}[FP]{c.RESET}  "
            elif tag == "FN":
                tag_str = f"{c.BOLD}{c.BRIGHT_YELLOW}[FN]{c.RESET}  "
            else:
                tag_str = f"[{tag}]"

            if ": " in body:
                stmt, reason = body.split(": ", 1)
                line = f"{indent}- {tag_str} {pipe} {stmt.strip()} {pipe} {c.DIM}Alasan:{c.RESET} {reason.strip()}"
            else:
                line = f"{indent}- {tag_str} {pipe} {body}"
            formatted.append(line)
        else:
            formatted.append(f"{indent}- {item}")

    return formatted


def format_sample_evaluation(
    idx: int,
    total: int,
    res: dict,
    duration: float,
) -> str:
    """
    Membangun tampilan log lengkap per sampel evaluasi:
    - ID Query & Pertanyaan
    - Skor Metrik (dengan warna)
    - Output Reasoning terperinci untuk Faithfulness, Context Recall, dan Answer Correctness
    """
    c = AnsiColor
    sep = f"{c.CYAN}{'=' * 80}{c.RESET}"
    sub_sep = f"{c.CYAN}{'-' * 80}{c.RESET}"

    qid = res.get("query_id", f"query_{idx}")
    qtext = res.get("query_text", "")
    if len(qtext) > 100:
        qtext_summary = qtext[:97] + "..."
    else:
        qtext_summary = qtext

    f_score = colorize_score(res.get("faithfulness_score"))
    cr_score = colorize_score(res.get("context_recall_score"))
    ac_score = colorize_score(res.get("answer_correctness_score"))
    sim_score = colorize_score(res.get("answer_correctness_similarity"))
    f1_score = colorize_score(res.get("answer_correctness_f1"))

    lines = [
        sep,
        f"{c.BOLD}{c.BRIGHT_CYAN}[EVALUASI SAMPEL {idx}/{total}]{c.RESET} "
        f"{c.BOLD}Query ID:{c.RESET} {c.BRIGHT_WHITE}{qid}{c.RESET} | "
        f"{c.DIM}Durasi:{c.RESET} {c.YELLOW}{duration:.2f}s{c.RESET}",
        f"  {c.BOLD}Pertanyaan :{c.RESET} {qtext_summary}",
        f"  {c.BOLD}Skor Metrik:{c.RESET} "
        f"Faithfulness: {f_score} | "
        f"Context Recall: {cr_score} | "
        f"Answer Correctness: {ac_score} (Sim: {sim_score}, F1: {f1_score})",
        sub_sep,
    ]

    # 1. Faithfulness Reasoning
    f_reasoning = res.get("faithfulness_reasoning", "")
    lines.append(f"  {c.BOLD}{c.BRIGHT_MAGENTA}[REASONING FAITHFULNESS]{c.RESET} (Skor: {f_score})")
    lines.extend(format_reasoning_items(f_reasoning, indent="    "))
    lines.append(sub_sep)

    # 2. Context Recall Reasoning
    cr_reasoning = res.get("context_recall_reasoning", "")
    lines.append(f"  {c.BOLD}{c.BRIGHT_BLUE}[REASONING CONTEXT RECALL]{c.RESET} (Skor: {cr_score})")
    lines.extend(format_reasoning_items(cr_reasoning, indent="    "))
    lines.append(sub_sep)

    # 3. Answer Correctness Reasoning
    ac_reasoning = res.get("answer_correctness_reasoning", "")
    lines.append(
        f"  {c.BOLD}{c.BRIGHT_CYAN}[REASONING ANSWER CORRECTNESS]{c.RESET} (Skor: {ac_score} | F1: {f1_score} | Sim: {sim_score})"
    )
    lines.extend(format_reasoning_items(ac_reasoning, indent="    "))

    lines.append(sep)
    return "\n".join(lines)


def setup_logger(model_name: str, custom_log_path: str = None) -> t.Tuple[logging.Logger, str]:
    """
    Menyiapkan logger terminal berwarna (ANSI) dan file log plain text (ANSI-stripped).
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    logs_dir = os.path.join(root_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    if not custom_log_path:
        sanitized_model = model_name.replace(":", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"eval_{sanitized_model}_{timestamp}.log")
    else:
        log_file = custom_log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

    logger = logging.getLogger(f"eval_{model_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File Handler (Plain text, auto-strips ANSI)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = PlainFileFormatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler (ANSI Colored)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredConsoleFormatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger, log_file
