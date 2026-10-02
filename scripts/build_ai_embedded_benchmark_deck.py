#!/usr/bin/env python3
"""Build the updated AI embedded report deck from the current PDF and benchmarks."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF = Path("/home/kth/Downloads/ai임베디드_보고서.pdf")
SOURCE_PPTX = Path("/home/kth/Documents/카카오톡 받은 파일/ai임베디드_보고서.pptx")
OUT_PPTX = ROOT / "ai임베디드_보고서_벤치마크반영.pptx"
DOWNLOAD_PPTX = Path("/home/kth/Downloads/ai임베디드_보고서_벤치마크반영.pptx")
EDITABLE_PPTX = Path("/home/kth/Downloads/ai임베디드_보고서_벤치마크반영_윈도우편집가능.pptx")

W, H = 20.0, 11.25
NAVY = "173F76"
NAVY_DARK = "102D54"
BLUE = "2F65A7"
LIGHT_BLUE = "EAF1FA"
YELLOW = "FFD633"
GREEN = "1A9B78"
ORANGE = "E39119"
RED = "D85858"
BLACK = "111111"
GRAY = "777777"
LIGHT_GRAY = "D9DDE3"
PALE = "F5F7FA"
WHITE = "FFFFFF"
FONT = "Malgun Gothic"


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def set_bg(slide, color: str = WHITE) -> None:
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(color)


def rect(slide, x, y, w, h, fill, radius=False, line=None, line_width=1):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    if line:
        shape.line.color.rgb = rgb(line)
        shape.line.width = Pt(line_width)
    else:
        shape.line.fill.background()
    return shape


def textbox(
    slide,
    x,
    y,
    w,
    h,
    content,
    size=24,
    color=BLACK,
    bold=False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin=0,
    font=FONT,
):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(margin)
    lines = str(content).split("\n")
    for idx, line in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.font.name = font
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = rgb(color)
        p.space_after = Pt(0)
    return shape


def line(slide, x1, y1, x2, y2, color=LIGHT_GRAY, width=1.3):
    shape = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    shape.line.color.rgb = rgb(color)
    shape.line.width = Pt(width)
    return shape


def arrow(slide, x1, y1, x2, y2, color=NAVY, width=2.5):
    shape = line(slide, x1, y1, x2, y2, color, width)
    shape.line.end_arrowhead = True
    return shape


def extract_logo(tmp: Path) -> Path:
    src = Presentation(SOURCE_PPTX)
    for slide in src.slides:
        for shape in slide.shapes:
            if shape.shape_type == 13 and shape.top < Inches(2.2) and shape.left > Inches(16):
                target = tmp / f"logo.{shape.image.ext}"
                target.write_bytes(shape.image.blob)
                return target
    raise RuntimeError("Could not find the university logo in source PPTX")


def add_logo(slide, logo: Path) -> None:
    slide.shapes.add_picture(str(logo), Inches(17.45), Inches(1.64), width=Inches(1.37))


def header(slide, logo: Path, number: str, title: str, subtitle: str | None = None, page=1):
    set_bg(slide)
    textbox(
        slide, 1.18, 1.05, 11.2, 0.4,
        "온디바이스 AI를 활용한 개인 맞춤형 자세 감지·음성 코칭 시스템",
        13, "B9B9B9", True,
    )
    textbox(slide, 1.18, 1.52, 1.05, 0.72, number, 31, NAVY, True)
    rect(slide, 1.16, 2.37, 0.9, 0.07, NAVY)
    textbox(slide, 2.25, 1.58, 10.6, 0.72, title, 26, BLACK, True)
    if subtitle:
        textbox(slide, 12.3, 1.72, 4.7, 0.45, subtitle, 14, GRAY, True, PP_ALIGN.RIGHT)
    line(slide, 2.05, 2.42, 18.8, 2.42, "BFC3C8", 1.2)
    add_logo(slide, logo)
    textbox(slide, 19.35, 10.45, 0.48, 0.42, str(page), 15, "A6A6A6", True, PP_ALIGN.RIGHT)


def bullet_text(slide, x, y, w, lines, size=17, color=BLACK, gap=0.58):
    for idx, item in enumerate(lines):
        textbox(slide, x, y + idx * gap, 0.3, 0.35, "•", size, NAVY, True)
        textbox(slide, x + 0.34, y + idx * gap, w - 0.34, gap, item, size, color, False)


def pill(slide, x, y, w, text_value, fill=LIGHT_BLUE, color=NAVY):
    rect(slide, x, y, w, 0.46, fill, True)
    textbox(slide, x, y + 0.02, w, 0.4, text_value, 13, color, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)


def crop_reference(page: Path, target: Path, crop_box: tuple[float, float, float, float]) -> Path:
    """Crop a normalized (left, top, right, bottom) box from a rendered page."""
    from PIL import Image

    with Image.open(page) as image:
        width, height = image.size
        left, top, right, bottom = crop_box
        image.crop((left * width, top * height, right * width, bottom * height)).save(target)
    return target


def add_cover_slide(prs, logo, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    slide.shapes.add_picture(str(logo), Inches(8.85), Inches(1.9), width=Inches(2.35))
    rect(slide, 9.08, 4.15, 1.84, 0.06, NAVY)
    textbox(slide, 2.7, 4.9, 14.6, 0.9,
            "온디바이스 AI를 활용한 개인 맞춤형 자세 감지 및 음성 코칭 시스템",
            30, BLACK, True, PP_ALIGN.CENTER)
    textbox(slide, 6.55, 6.0, 6.9, 0.55, "AI임베디드 주제 발표 보고서", 18, "8A8A8A", True, PP_ALIGN.CENTER)
    rect(slide, 0, 9.35, 20, 1.9, NAVY_DARK)
    textbox(slide, 14.1, 9.92, 5.25, 0.4, "김태현 / 탁윤성 / 변대필 / 김민석 / 김찬영", 14, WHITE, True, PP_ALIGN.RIGHT)
    textbox(slide, 16.0, 10.39, 3.35, 0.45, "1조 김태현과 아이들", 18, WHITE, True, PP_ALIGN.RIGHT)


def add_team_slide(prs, logo, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "01", "팀 소개", page=page)
    members = [
        ("김태현", 7.25, 2.72),
        ("탁윤성", 1.72, 5.55),
        ("변대필", 12.75, 5.55),
        ("김민석", 4.2, 8.45),
        ("김찬영", 10.25, 8.45),
    ]
    for name, x, y in members:
        rect(slide, x, y, 5.5, 2.02, "F1F3FF", True, "ECEEF5", 0.8)
        textbox(slide, x, y + 0.22, 5.5, 0.58, name, 22, BLACK, True, PP_ALIGN.CENTER)
        textbox(slide, x + 0.35, y + 1.0, 4.8, 0.45, "역할 및 학번 입력", 14, "B7BAC3", False, PP_ALIGN.CENTER)
    star = slide.shapes.add_shape(MSO_SHAPE.STAR_5_POINT, Inches(9.6), Inches(6.55), Inches(0.65), Inches(0.65))
    star.fill.solid(); star.fill.fore_color.rgb = rgb(YELLOW); star.line.color.rgb = rgb(NAVY_DARK)


def add_contents_slide(prs, logo, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_logo(slide, logo)
    textbox(slide, 6.9, 1.7, 6.2, 0.8, "C O N T E N T S", 30, BLACK, True, PP_ALIGN.CENTER)
    items = [
        ("1. 프로젝트 개요", "p.04"),
        ("2. 연구 배경", "p.07"),
        ("3. 프로젝트 동기", "p.10"),
        ("4. 개발 현황·방향", "p.13"),
    ]
    for i, (name, ref) in enumerate(items):
        y = 4.0 + i * 1.2
        textbox(slide, 6.82, y, 5.2, 0.55, name, 22, BLACK, True)
        textbox(slide, 12.25, y + 0.18, 0.8, 0.3, ref, 12, GRAY, True, PP_ALIGN.RIGHT)
        rect(slide, 6.82, y + 0.61, 6.25, 0.04, NAVY)


def add_topic_divider(prs, logo, number, title, bullets, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    rect(slide, 9.98, 0, 10.02, 11.25, NAVY_DARK)
    rect(slide, 10.35, 0, 9.65, 11.25, "101B60")
    textbox(slide, 1.15, 1.28, 8.0, 0.5,
            "온디바이스 AI를 활용한 개인 맞춤형 자세 감지·음성 코칭 시스템",
            14, "C0C0C0", True)
    textbox(slide, 1.15, 1.9, 7.9, 0.72, title, 30, BLACK, True)
    if bullets:
        bullet_text(slide, 1.15, 3.24, 7.5, bullets, 17, GRAY, gap=0.52)
    textbox(slide, 1.15, 9.95, 4.5, 0.45, "AI임베디드 주제 발표 보고서", 14, "52607A", True)
    textbox(slide, 12.0, 6.0, 7.2, 4.1, number, 104, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    slide.shapes.add_picture(str(logo), Inches(18.28), Inches(0.2), width=Inches(1.45))
    textbox(slide, 19.36, 10.47, 0.45, 0.4, str(page), 14, "D8DDE5", True, PP_ALIGN.RIGHT)


def add_overview_title_slide(prs, logo, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "01", "프로젝트 개요", page=page)
    rect(slide, 3.0, 5.72, 3.05, 0.56, YELLOW, True)
    rect(slide, 8.02, 5.72, 4.4, 0.56, YELLOW, True)
    textbox(slide, 2.35, 5.55, 15.35, 1.0,
            "“ 온디바이스 AI를 활용한 개인 맞춤형 자세 감지 및 음성 코칭 시스템 ”",
            27, BLACK, True, PP_ALIGN.CENTER)
    textbox(slide, 5.4, 6.92, 9.2, 0.48,
            "디지털 헬스 / 비접촉 자세 모니터링 / 임베디드 AI 자세 교정 지원",
            17, "BBBBBB", True, PP_ALIGN.CENTER)


def add_overview_bullets_slide(prs, logo, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "01", "프로젝트 개요", page=page)
    textbox(slide, 5.2, 3.05, 10.0, 0.6, "장시간 앉아있는 사람들을 위한 자세 교정 코치", 26, BLACK, True, PP_ALIGN.CENTER)
    bullet_text(slide, 3.2, 4.2, 14.3, [
        "카메라를 통해 자세를 확인",
        "Jetson Nano에서 현재 자세가 올바른지 아닌지 분석",
        "현재 자세가 좋지 못할 때 스피커 또는 경고문으로 알림",
        "장시간 앉아있을 경우 스트레칭 권고로 피로 누적 해소",
    ], 22, gap=1.02)


def add_background_quote_slide(prs, logo, page, article_img):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "02", "연구 배경 - 잘못된 좌식 습관의 경고", page=page)
    slide.shapes.add_picture(str(article_img), Inches(0.65), Inches(4.0), width=Inches(6.9), height=Inches(4.9))
    textbox(slide, 8.05, 4.02, 10.6, 1.4,
            "“거북목 자세가 오래 지속되면 목 뼈 주변의 근육·인대·디스크에 부담이 누적될 수 있습니다.”",
            21, BLACK, True, PP_ALIGN.CENTER)
    textbox(slide, 12.0, 5.48, 3.0, 0.45, "국민건강지식센터", 18, "C5C5C5", True, PP_ALIGN.CENTER)
    textbox(slide, 8.05, 6.42, 10.6, 1.65,
            "“앉는 순간 골반이 뒤로 밀리면서 허리의 정상적인 곡선이 무너지기 쉽고, 디스크 압력을 높일 수 있습니다.”",
            21, BLACK, True, PP_ALIGN.CENTER)
    textbox(slide, 12.75, 8.18, 1.6, 0.45, "헬스조선", 18, "C5C5C5", True, PP_ALIGN.CENTER)


def add_sitting_stats_slide(prs, logo, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "02", "연구 배경 - 좌식 행동으로 보내는 하루 평균 시간", page=page)
    textbox(slide, 1.0, 3.55, 7.0, 0.5, "국내 성인의 하루 평균 좌식 시간", 21, NAVY_DARK, True)
    textbox(slide, 1.0, 4.2, 4.0, 0.35, "2023년 국민건강통계", 13, "647086", True)
    labels = ["전체 (19세 이상)", "19~29세", "30~39세", "40~49세", "50~59세"]
    values = [9.0, 9.9, 9.4, 9.0, 8.5]
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 4.72 + i * 0.75
        textbox(slide, 1.15, y + 0.08, 2.0, 0.38, label, 15, NAVY_DARK, False, PP_ALIGN.RIGHT)
        rect(slide, 3.25, y, value * 0.52, 0.46, "198A92" if i == 0 else BLUE)
        textbox(slide, 3.25 + value * 0.52 + 0.2, y + 0.04, 1.0, 0.38, f"{value:.1f}시간", 14, NAVY_DARK, True)
    textbox(slide, 10.75, 3.55, 7.0, 0.5, "사무직의 근무시간 구성", 21, NAVY_DARK, True)
    textbox(slide, 10.75, 4.2, 7.0, 0.35, "정부기관 사무직 229명 · 신체 부착형 기기로 측정", 13, "647086", True)
    x, y, total_w = 10.9, 5.28, 7.65
    parts = [(42.1, "3868AF", "42.1%", WHITE), (36.7, "83A9DF", "36.7%", BLACK), (21.2, "D8DFE8", "21.2%", BLACK)]
    cur = x
    for val, color, label, txt_color in parts:
        width = total_w * val / 100
        rect(slide, cur, y, width, 1.0, color)
        textbox(slide, cur, y + 0.22, width, 0.52, label, 24, txt_color, False, PP_ALIGN.CENTER)
        cur += width
    textbox(slide, 12.6, 6.62, 4.2, 0.48, "전체 앉은 시간 78.8%", 21, NAVY, True, PP_ALIGN.CENTER)
    rect(slide, 10.9, 7.35, 7.65, 0.95, LIGHT_BLUE, True)
    textbox(slide, 10.9, 7.58, 7.65, 0.52, "8시간 근무 기준 약 6.3시간 앉음", 23, BLUE, True, PP_ALIGN.CENTER)


def add_motivation_slide(prs, logo, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "03", "프로젝트 동기 - 연구 필요성", page=page)
    textbox(slide, 5.8, 3.45, 6.7, 0.6, "성인의 좌식 시간 추이", 27, BLACK, True, PP_ALIGN.CENTER)
    chart_x, chart_y, chart_w, chart_h = 1.55, 5.05, 7.7, 3.2
    for tick in [0, 20, 40, 60, 80]:
        yy = chart_y + chart_h - chart_h * tick / 80
        line(slide, chart_x, yy, chart_x + chart_w, yy, "DADDE2", 0.8)
        textbox(slide, chart_x - 0.55, yy - 0.18, 0.45, 0.35, str(tick), 11, GRAY, False, PP_ALIGN.RIGHT)
    years = [2014, 2017, 2021]
    vals = [46.7, 56.2, 63.0]
    points = []
    for i, (yr, val) in enumerate(zip(years, vals)):
        x = chart_x + 1.25 + i * 2.55
        y = chart_y + chart_h - chart_h * val / 80
        rect(slide, x - 0.62, y, 1.24, chart_y + chart_h - y, "F7C6C6")
        dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x - 0.07), Inches(y - 0.07), Inches(0.14), Inches(0.14))
        dot.fill.solid(); dot.fill.fore_color.rgb = rgb("F26B6B"); dot.line.fill.background()
        textbox(slide, x - 0.5, y - 0.48, 1.0, 0.35, f"{val:g}", 13, "F26B6B", False, PP_ALIGN.CENTER)
        textbox(slide, x - 0.5, chart_y + chart_h + 0.12, 1.0, 0.35, str(yr), 12, BLACK, False, PP_ALIGN.CENTER)
        points.append((x, y))
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        line(slide, x1, y1, x2, y2, "F26B6B", 1.2)
    textbox(slide, 3.65, 8.72, 4.0, 0.35, "● 8시간 이상 좌식생활하는 성인 비율", 12, BLACK, False, PP_ALIGN.CENTER)
    textbox(slide, 10.3, 5.85, 8.2, 1.5,
            "앉아서 일하는 성인의 비율은 해마다 늘어나고 있고\n흔해진 만큼 자세에 대한 인식이 중요해지고 있다.",
            22, BLACK, True, PP_ALIGN.CENTER)


def add_market_slide(prs, logo, page, market_img):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "03", "프로젝트 동기 - 시장성", page=page)
    textbox(slide, 1.0, 2.75, 2.0, 0.45, "시장성", 18, BLACK, True)
    textbox(slide, 6.7, 2.85, 7.9, 0.7, "시장성 전망(2025 ~ 2030년)", 27, BLACK, True, PP_ALIGN.CENTER)
    slide.shapes.add_picture(str(market_img), Inches(1.05), Inches(3.65), width=Inches(8.1), height=Inches(5.4))
    textbox(slide, 9.75, 4.0, 9.2, 2.15,
            "Grand View Research에 따르면 자세 교정 시장의\n2025~2030년 예상 연평균 성장률은 8.4%이며,\n2030년 약 20억 달러 규모로 성장할 것으로 예상된다.",
            20, BLACK, True)
    textbox(slide, 9.75, 6.7, 8.9, 1.3,
            "이는 자세 교정 제품의 시장 기회가\n확대되고 있음을 보여준다.",
            21, BLACK, True)
    textbox(slide, 5.5, 9.25, 4.0, 0.45, "grand view research", 20, "A7A7A7", True, PP_ALIGN.CENTER)


def add_editable_legacy_pages(prs: Presentation, rendered_pages: list[Path], logo: Path, tmp: Path) -> None:
    article = crop_reference(rendered_pages[7], tmp / "article.png", (0.015, 0.31, 0.39, 0.83))
    market = crop_reference(rendered_pages[11], tmp / "market.png", (0.035, 0.32, 0.455, 0.85))
    add_cover_slide(prs, logo, 1)
    add_team_slide(prs, logo, 2)
    add_contents_slide(prs, logo, 3)
    add_topic_divider(prs, logo, "01", "프로젝트 개요", [], 4)
    add_overview_title_slide(prs, logo, 5)
    add_overview_bullets_slide(prs, logo, 6)
    add_topic_divider(prs, logo, "02", "연구 배경", ["잘못된 좌식 습관의 경고", "좌식 행동으로 보내는 하루 평균 시간"], 7)
    add_background_quote_slide(prs, logo, 8, article)
    add_sitting_stats_slide(prs, logo, 9)
    add_topic_divider(prs, logo, "03", "프로젝트 동기", ["연구 필요성", "시장성"], 10)
    add_motivation_slide(prs, logo, 11)
    add_market_slide(prs, logo, 12, market)


def add_section_slide(prs, logo, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    rect(slide, 9.22, 0.62, 10.78, 10.19, NAVY_DARK)
    rect(slide, 10.0, 0.62, 10.0, 10.19, NAVY)
    textbox(slide, 1.15, 1.26, 8.0, 0.46,
            "온디바이스 AI를 활용한 개인 맞춤형 자세 감지·음성 코칭 시스템",
            13, "B9B9B9", True)
    textbox(slide, 1.15, 1.72, 7.4, 0.9, "개발 현황·병목 분석·향후 계획", 29, BLACK, True)
    textbox(slide, 1.15, 10.02, 4.4, 0.48, "AI임베디드 주제 발표 보고서", 12, GRAY)
    textbox(slide, 12.2, 5.7, 7.1, 3.0, "04", 90, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    add_logo(slide, logo)
    textbox(slide, 19.35, 10.45, 0.48, 0.42, str(page), 15, "D8DDE5", True, PP_ALIGN.RIGHT)


def add_mvp_slide(prs, logo, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "04", "현재 개발 현황 - 온디바이스 MVP", "어제 실측 기준", page)
    stages = [
        ("카메라 입력", "CSI 카메라\n측면 착석 영상", True),
        ("자세 추론", "Pose-ResNet18\nTensorRT FP16", True),
        ("개인화 판정", "귀·어깨·엉덩이\n기준·지속시간", True),
        ("시각화·계측", "웹 대시보드\n벤치마크 로그", True),
        ("피드백 통합", "장치 오디오\nGPIO·Bluetooth", False),
    ]
    x0, y, w, gap = 1.2, 3.2, 3.25, 0.55
    for i, (name, body, done) in enumerate(stages):
        x = x0 + i * (w + gap)
        rect(slide, x, y, w, 3.0, WHITE, True, GREEN if done else ORANGE, 2.2)
        pill(slide, x + 0.62, y + 0.34, 2.0, "구현 완료" if done else "통합 예정", GREEN if done else ORANGE, WHITE)
        textbox(slide, x + 0.2, y + 1.04, w - 0.4, 0.55, name, 20, BLACK, True, PP_ALIGN.CENTER)
        textbox(slide, x + 0.25, y + 1.75, w - 0.5, 0.9, body, 16, GRAY, False, PP_ALIGN.CENTER)
        if i < len(stages) - 1:
            arrow(slide, x + w + 0.08, y + 1.5, x + w + gap - 0.08, y + 1.5, NAVY, 2.4)
    rect(slide, 1.2, 7.05, 17.6, 2.15, PALE, True)
    textbox(slide, 1.55, 7.35, 3.1, 0.48, "현재 달성", 18, NAVY, True)
    bullet_text(slide, 1.55, 7.93, 7.2, [
        "카메라 → 고정 ROI → GPU 추론 → 개인 기준 판정 → 대시보드",
        "원본 영상 외부 전송 없이 Jetson에서 로컬 처리",
    ], 16, gap=0.55)
    textbox(slide, 10.05, 7.35, 3.1, 0.48, "남은 통합", 18, ORANGE, True)
    bullet_text(slide, 10.05, 7.93, 7.9, [
        "음성 파일 재생, 착석 센서·LED·버튼, 상태 전송",
        "사용자 데이터 평가와 30분 이상 연속 실행 검증",
    ], 16, gap=0.55)


def add_model_slide(prs, logo, page, model_img):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "04", "모델 단독 성능 - 경량화 목표 통과", "Pose-ResNet18 · TensorRT FP16", page)
    metrics = [
        ("13.10", "FPS", "목표 ≥ 5", GREEN),
        ("78.30", "ms", "추론 p95 ≤ 200ms", GREEN),
        ("1.41", "GB", "RSS peak ≤ 3.5GB", GREEN),
        ("48.5", "°C", "최고 온도 < 80°C", GREEN),
    ]
    for i, (value, unit, target, color) in enumerate(metrics):
        x = 1.2 + (i % 2) * 4.1
        y = 3.0 + (i // 2) * 2.35
        rect(slide, x, y, 3.65, 1.9, PALE, True, LIGHT_GRAY)
        textbox(slide, x + 0.28, y + 0.24, 2.15, 0.7, value, 31, NAVY, True)
        textbox(slide, x + 2.38, y + 0.39, 0.85, 0.4, unit, 16, GRAY, True)
        pill(slide, x + 0.28, y + 1.17, 2.95, target, color, WHITE)
    rect(slide, 9.65, 2.85, 8.95, 5.95, NAVY_DARK, True)
    slide.shapes.add_picture(str(model_img), Inches(9.87), Inches(3.02), width=Inches(8.5), height=Inches(4.78))
    textbox(slide, 10.0, 8.05, 8.2, 0.45, "60초 모델 단독 측정 · 오류 0건", 14, WHITE, True, PP_ALIGN.CENTER)
    rect(slide, 1.2, 8.1, 7.75, 0.72, LIGHT_BLUE, True)
    textbox(slide, 1.45, 8.24, 7.25, 0.42,
            "결론: 자세 추론 모델 자체는 실시간 처리 가능",
            17, NAVY, True, PP_ALIGN.CENTER)
    textbox(slide, 1.2, 9.25, 17.4, 0.42,
            "※ 합성 프레임 기준. 카메라·얼굴 검출·네트워크는 포함하지 않음.",
            13, GRAY)


def add_test_design_slide(prs, logo, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "04", "어제 수행한 테스트 - 목적·조건·측정 지표", "2026.09.28 · Jetson Nano 4GB", page)
    tests = [
        ("① 모델 단독", "TensorRT FP16 엔진\n합성 프레임 · 60초", "FPS · 추론 p95\nRSS · 온도 · 오류", "모델 자체의\n엣지 실행 가능성"),
        ("② 실시간 파이프라인", "CSI 카메라 → 얼굴 ROI\n→ 자세 추론 → 웹", "전체 FPS · 지연 p95\n얼굴 검출 · 추론 시간", "불필요한 얼굴 검출을\n제거할 근거 확인"),
        ("③ 자세 조건 비교", "손: 허벅지 / 가슴 높이\n등받이: 기대기 / 비기대", "raw 유효 자세율\n엉덩이 confidence · 상태 전환", "어떤 착석 조건에서\n판정이 흔들리는지"),
        ("④ 네트워크 RTT", "CAC-5G · 5GHz · -47dBm\n20Hz · 30초 · 오류 0건", "HTTP RTT p50 / p95\n촬영→클라이언 수신 지연", "Wi-Fi가 3.5 FPS의\n주요 원인인지"),
    ]
    for i, (name, condition, metric, question) in enumerate(tests):
        x = 1.2 + (i % 2) * 8.95
        y = 3.0 + (i // 2) * 3.15
        rect(slide, x, y, 8.45, 2.68, WHITE, True, LIGHT_GRAY, 1.25)
        textbox(slide, x + 0.34, y + 0.26, 3.45, 0.48, name, 19, NAVY, True)
        textbox(slide, x + 0.34, y + 0.88, 3.6, 1.02, condition, 15, BLACK, True)
        line(slide, x + 4.12, y + 0.3, x + 4.12, y + 2.35, LIGHT_GRAY, 1)
        textbox(slide, x + 4.45, y + 0.3, 3.55, 0.35, "측정", 13, GRAY, True)
        textbox(slide, x + 4.45, y + 0.78, 3.55, 0.78, metric, 14, BLACK, True)
        textbox(slide, x + 4.45, y + 1.72, 3.55, 0.65, question, 13, NAVY, True)
    rect(slide, 1.2, 9.55, 17.6, 0.55, "FFF5C7", True)
    textbox(slide, 1.45, 9.65, 17.1, 0.34,
            "공통 조건: 워밍업 5초 후 자세별 60초 · 키포인트 confidence ≥ 0.25 · 참가자 1명의 탐색 실험",
            14, BLACK, True, PP_ALIGN.CENTER)


def add_pipeline_slide(prs, logo, page, rtt_img):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "04", "전체 파이프라인 - 불필요한 얼굴 검출이 병목", "5개 자세 조건 · 각 60초", page)
    textbox(slide, 1.2, 2.85, 9.5, 0.58, "프레임 처리 시간(대표값, 약 303ms)", 20, BLACK, True)
    total_x, total_y, total_w, bar_h = 1.2, 3.75, 10.0, 1.05
    pieces = [
        ("Face detect\n201~205ms", 203, NAVY),
        ("Pose\n≈69ms", 69, YELLOW),
        ("기타\n≈31ms", 31, "BFC7D2"),
    ]
    cur = total_x
    for label, value, color in pieces:
        width = total_w * value / 303
        rect(slide, cur, total_y, width, bar_h, color)
        textbox(slide, cur, total_y + 0.13, width, 0.72, label, 15,
                WHITE if color == NAVY else BLACK, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
        cur += width
    line(slide, total_x, 5.25, total_x + total_w, 5.25, LIGHT_GRAY, 1)
    metric_rows = [
        ("전체 FPS", "3.53 ~ 3.58", "목표 5 FPS 미달", RED),
        ("전체 지연 p95", "316.85 ~ 322.11ms", "목표 200ms 초과", RED),
        ("HTTP RTT p95", "26.94ms", "네트워크는 주요 원인 아님", GREEN),
    ]
    for i, (name, value, note, color) in enumerate(metric_rows):
        y = 5.7 + i * 1.0
        textbox(slide, 1.2, y, 2.7, 0.46, name, 16, GRAY, True)
        textbox(slide, 4.0, y - 0.06, 3.2, 0.58, value, 21, NAVY, True)
        pill(slide, 7.2, y - 0.02, 3.7, note, color, WHITE)
    rect(slide, 12.0, 2.85, 6.8, 5.85, NAVY_DARK, True)
    slide.shapes.add_picture(str(rtt_img), Inches(12.2), Inches(3.08), width=Inches(6.4), height=Inches(3.6))
    textbox(slide, 12.38, 7.0, 6.05, 1.25,
            "촬영→클라이언트 p95 354.77ms\nRTT의 약 13배",
            20, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    rect(slide, 1.2, 9.05, 17.6, 0.72, "FFF5C7", True)
    textbox(slide, 1.4, 9.2, 17.2, 0.38,
            "실험 결론: 얼굴 인식은 불필요 → 얼굴 검출 제거 후 고정 ROI로 대체",
            18, BLACK, True, PP_ALIGN.CENTER)


def add_reliability_slide(prs, logo, page, good_img, fail_img):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "04", "자세 판정 신뢰성 - 엉덩이 가시성이 제한", "confidence ≥ 0.25", page)
    labels = ["허벅지\n비기대", "손 들기\n비기대 1", "손 들기\n기대", "손 들기\n비기대 2", "허벅지\n기대"]
    values = [65.12, 86.05, 85.45, 28.77, 100.0]
    chart_x, chart_y, chart_w, chart_h = 1.2, 3.35, 9.25, 4.9
    for tick in [0, 25, 50, 75, 100]:
        yy = chart_y + chart_h - chart_h * tick / 100
        line(slide, chart_x + 0.7, yy, chart_x + chart_w, yy, "E1E4E8", 0.8)
        textbox(slide, chart_x, yy - 0.18, 0.55, 0.35, str(tick), 12, GRAY, False, PP_ALIGN.RIGHT)
    bar_w = 1.18
    spacing = 1.68
    for i, (lab, value) in enumerate(zip(labels, values)):
        x = chart_x + 1.0 + i * spacing
        h = chart_h * value / 100
        color = NAVY if i != 3 else RED
        rect(slide, x, chart_y + chart_h - h, bar_w, h, color, True)
        textbox(slide, x - 0.2, chart_y + chart_h - h - 0.5, bar_w + 0.4, 0.4,
                f"{value:.2f}%", 14, color, True, PP_ALIGN.CENTER)
        textbox(slide, x - 0.27, chart_y + chart_h + 0.15, bar_w + 0.54, 0.72,
                lab, 11, GRAY, True, PP_ALIGN.CENTER)
    textbox(slide, 1.2, 2.78, 9.0, 0.45, "raw 유효 자세율", 18, BLACK, True)
    rect(slide, 11.2, 2.95, 7.6, 2.65, WHITE, True, LIGHT_GRAY)
    slide.shapes.add_picture(str(good_img), Inches(11.38), Inches(3.13), width=Inches(3.45), height=Inches(2.3))
    slide.shapes.add_picture(str(fail_img), Inches(15.18), Inches(3.13), width=Inches(3.45), height=Inches(2.3))
    textbox(slide, 11.35, 5.72, 7.3, 0.42, "같은 장소에서도 골반 위치·가림에 따라 판정 가능 여부가 바뀌", 14, GRAY, True, PP_ALIGN.CENTER)
    rect(slide, 11.2, 6.45, 7.6, 2.95, PALE, True)
    textbox(slide, 11.55, 6.78, 2.45, 0.45, "핵심 관찰", 18, NAVY, True)
    bullet_text(slide, 11.55, 7.36, 6.65, [
        "빨간 점: confidence 0.25 미만, 판정에 미사용",
        "100%는 정확도가 아닌 필수 관절 판정 가능률",
        "허벅지·비기대는 65.12% → 손 위치 효과 확정 불가",
    ], 15, gap=0.62)
    textbox(slide, 1.2, 9.65, 17.6, 0.4,
            "※ 1명·조건별 1회의 탐색 실험으로, 손·등받이의 인과효과를 확정하지 않음.",
            13, GRAY)


def add_diagnosis_slide(prs, logo, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "04", "조건별 병목 - 모델·파이프라인·자세의 문제를 분리", "실측 근거 기반", page)
    rows = [
        ("모델 단독", "13.10 FPS · p95 78.30ms", "PASS", "모델은 병목 아님"),
        ("전체 실시간 5조건", "3.53~3.58 FPS · p95 317~322ms", "속도", "불필요 Face detect 201~205ms"),
        ("손 들기·비기대 반복", "유효률 86.05% → 28.77%", "재현성", "골반 위치·가림에 민감"),
        ("손 들기·등받이 기대기", "유효률 85.45% · 59.4회/분", "안정성", "임계값 근처 상태 진동"),
        ("허벅지·등받이 기대기", "유효률 100% · 0회/분", "양호", "유리한 1회 조건, 일반화 불가"),
    ]
    headers = [("테스트 조건", 1.2, 5.3), ("관찰 결과", 6.5, 5.15), ("판정", 11.65, 2.0), ("해석된 병목", 13.65, 5.15)]
    for label, x, w in headers:
        rect(slide, x, 2.92, w, 0.62, NAVY)
        textbox(slide, x, 3.04, w, 0.34, label, 14, WHITE, True, PP_ALIGN.CENTER)
    for idx, (condition, result, verdict, bottleneck) in enumerate(rows):
        y = 3.6 + idx * 1.16
        fill = WHITE if idx % 2 == 0 else PALE
        rect(slide, 1.2, y, 17.6, 1.02, fill)
        textbox(slide, 1.45, y + 0.24, 4.8, 0.47, condition, 15, BLACK, True)
        textbox(slide, 6.72, y + 0.24, 4.65, 0.47, result, 15, NAVY, True)
        verdict_color = GREEN if verdict in {"PASS", "양호"} else RED
        pill(slide, 11.85, y + 0.22, 1.62, verdict, verdict_color, WHITE)
        textbox(slide, 13.9, y + 0.24, 4.55, 0.47, bottleneck, 14, GRAY, True)
    rect(slide, 1.2, 9.65, 17.6, 0.58, "FFF5C7", True)
    textbox(slide, 1.45, 9.76, 17.1, 0.35,
            "결론: 현재 속도 병목은 불필요한 얼굴 검출 → 제거 후 고정 ROI 파이프라인으로 재측정",
            15, BLACK, True, PP_ALIGN.CENTER)


def add_research_slide(prs, logo, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "04", "향후 논문 탐색 - 실험 질문으로 연결", "성능 개선 + 판정 강건성", page)
    themes = [
        ("① 시계열 자세 추정", "가림·누락 키포인트를\n이전 프레임으로 보완할 수 있는가?", "temporal pose / pose tracking"),
        ("② 상체 전용 자세 분류", "엉덩이가 없을 때 머리 자세만\n안전하게 제한 판정할 수 있는가?", "upper-body classification"),
        ("③ 얼굴 검출 없는 ROI", "단일 사용자·고정 카메라에서\n고정 ROI로 즉시 추론할 수 있는가?", "fixed ROI / initial setup"),
        ("④ 불확실성 보정", "confidence 임계값 하나가 아니라\n누락 시간·흔들림을 같이 쓸 수 있는가?", "uncertainty / calibration"),
    ]
    for i, (title, question, keyword) in enumerate(themes):
        x = 1.2 + (i % 2) * 8.95
        y = 3.0 + (i // 2) * 3.05
        rect(slide, x, y, 8.45, 2.58, WHITE, True, LIGHT_GRAY, 1.2)
        textbox(slide, x + 0.35, y + 0.28, 7.7, 0.48, title, 19, NAVY, True)
        textbox(slide, x + 0.35, y + 0.91, 7.65, 0.92, question, 16, BLACK, True)
        pill(slide, x + 0.35, y + 1.95, 3.25, keyword, NAVY, WHITE)
    rect(slide, 1.2, 9.35, 17.6, 0.72, LIGHT_BLUE, True)
    textbox(slide, 1.48, 9.49, 17.0, 0.4,
            "기존 기준: Lee et al.(2023) 자세 분류 + Lite-HRNet(2021)·Lite Pose(2022) 경량화 설계",
            16, NAVY, True, PP_ALIGN.CENTER)


def add_roadmap_slide(prs, logo, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "04", "파이프라인 발전 계획 - 논문 → 구현 → 통제 실험", "단계별 성공 조건", page)
    steps = [
        ("STEP 1", "불필요 단계 제거", "얼굴 검출 제거\n고정 ROI·최초 설정", "≥ 5 FPS\np95 ≤ 200ms"),
        ("STEP 2", "누락 대응", "temporal fallback\n상체 제한 모드", "판정 가능률·\n오경보 동시 평가"),
        ("STEP 3", "통제 실험", "위치 표시·순서 무작위\n조건별 ≥3회 반복", "반복 편차 축소\n사용자 확장"),
        ("STEP 4", "하드웨어 통합", "TTS 오디오·GPIO\nBluetooth·로컬 기록", "30분 연속 실행\n피드백 효과 평가"),
    ]
    x_positions = [1.2, 5.7, 10.2, 14.7]
    for i, ((tag, title, body, criterion), x) in enumerate(zip(steps, x_positions)):
        pill(slide, x + 0.68, 3.0, 2.15, tag, NAVY if i < 3 else ORANGE, WHITE)
        rect(slide, x, 3.7, 3.65, 4.25, WHITE, True, LIGHT_GRAY, 1.4)
        textbox(slide, x + 0.2, 4.05, 3.25, 0.55, title, 20, BLACK, True, PP_ALIGN.CENTER)
        textbox(slide, x + 0.25, 4.9, 3.15, 1.0, body, 16, GRAY, True, PP_ALIGN.CENTER)
        line(slide, x + 0.42, 6.2, x + 3.23, 6.2, LIGHT_GRAY, 1)
        textbox(slide, x + 0.25, 6.52, 3.15, 0.88, criterion, 15, NAVY, True, PP_ALIGN.CENTER)
        if i < 3:
            arrow(slide, x + 3.78, 5.65, x + 4.38, 5.65, YELLOW, 4)
    rect(slide, 1.2, 8.65, 17.15, 1.05, "FFF5C7", True)
    textbox(slide, 1.55, 8.86, 16.45, 0.62,
            "핵심 방향: ‘작동하는 시제품’에서 ‘재현 가능하고 오경보를 설명할 수 있는 시스템’으로",
            18, BLACK, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)


def add_summary_slide(prs, logo, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, logo, "04", "결론 - 어제 실험이 다음 개발 순서를 바꾸었다", "현재 → 병목 → 다음 검증", page)
    cards = [
        ("현재", "온디바이스 MVP 구현", "13.10 FPS 모델\n개인 기준 판정\n웹 대시보드", GREEN),
        ("병목", "전체 처리와 골반 가시성", "전체 3.5 FPS\nFace detect ≈203ms\n반복 차이 57.28%p", RED),
        ("다음", "논문 기반 파이프라인 개선", "얼굴 검출 제거·고정 ROI\n시계열·상체 fallback\n통제 반복 실험", NAVY),
    ]
    for i, (tag, title, body, color) in enumerate(cards):
        x = 1.2 + i * 5.95
        rect(slide, x, 3.15, 5.35, 4.85, WHITE, True, color, 2)
        pill(slide, x + 1.38, 3.55, 2.6, tag, color, WHITE)
        textbox(slide, x + 0.35, 4.42, 4.65, 0.95, title, 20, BLACK, True, PP_ALIGN.CENTER)
        textbox(slide, x + 0.45, 5.72, 4.45, 1.6, body, 17, GRAY, True, PP_ALIGN.CENTER)
        if i < 2:
            arrow(slide, x + 5.42, 5.55, x + 5.86, 5.55, YELLOW, 4)
    rect(slide, 1.2, 8.78, 17.2, 0.85, LIGHT_BLUE, True)
    textbox(slide, 1.48, 8.97, 16.65, 0.47,
            "불필요한 얼굴 검출을 제거하고, ‘가림에 강한 자세 판정’에 연산을 집중",
            17, NAVY, True, PP_ALIGN.CENTER)


def main() -> None:
    if not SOURCE_PDF.exists():
        raise FileNotFoundError(SOURCE_PDF)
    if not SOURCE_PPTX.exists():
        raise FileNotFoundError(SOURCE_PPTX)

    with tempfile.TemporaryDirectory(prefix="ai-embedded-deck-") as td:
        tmp = Path(td)
        subprocess.run(
            ["pdftoppm", "-png", "-r", "120", str(SOURCE_PDF), str(tmp / "page")],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pages = sorted(tmp.glob("page-*.png"))
        if len(pages) < 12:
            raise RuntimeError(f"Expected at least 12 PDF pages, got {len(pages)}")
        logo = extract_logo(tmp)

        prs = Presentation()
        prs.slide_width = Inches(W)
        prs.slide_height = Inches(H)
        add_editable_legacy_pages(prs, pages, logo, tmp)
        add_section_slide(prs, logo, 13)
        add_mvp_slide(prs, logo, 14)
        add_test_design_slide(prs, logo, 15)
        add_model_slide(
            prs, logo, 16,
            ROOT / "captures/benchmark-20260928-cac5g-tts/11_model_benchmark_result.png",
        )
        add_pipeline_slide(
            prs, logo, 17,
            ROOT / "captures/benchmark-20260928-cac5g-tts/12_cac5g_network_rtt_result.png",
        )
        add_reliability_slide(
            prs, logo, 18,
            ROOT / "captures/benchmark-20260928-cac5g-tts/09_hands_on_thighs_leaning_backrest_start.png",
            ROOT / "captures/benchmark-20260928-cac5g-tts/07_hands_raised_no_backrest_repeat_start.png",
        )
        add_diagnosis_slide(prs, logo, 19)
        add_research_slide(prs, logo, 20)
        add_roadmap_slide(prs, logo, 21)
        add_summary_slide(prs, logo, 22)

        prs.save(OUT_PPTX)
        shutil.copy2(OUT_PPTX, DOWNLOAD_PPTX)
        shutil.copy2(OUT_PPTX, EDITABLE_PPTX)

    print(f"Created {len(prs.slides)} slides: {OUT_PPTX}")
    print(f"Copied to: {DOWNLOAD_PPTX}")
    print(f"Editable Windows copy: {EDITABLE_PPTX}")


if __name__ == "__main__":
    main()
