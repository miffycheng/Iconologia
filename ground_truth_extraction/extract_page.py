"""
extract_page.py
---------------
Extract ground-truth text from a single page of the ancient book XML,
identified by the image filename. Saves results as a JSON file.

Usage:
    python extract_page.py --image <image_path>

The script auto-detects the XML file from the image filename.
The output JSON is saved next to the image with the same base name.

Example:
    python extract_page.py \
        --image 1026A428_pdf_1-510_352_png.rf.e065fc1d114e613b0a8991da4101f184.jpg
"""

import re
import json
import argparse
import os
import xml.etree.ElementTree as ET


# ── Namespace used throughout the TEI XML ────────────────────────────────────
NS = {'tei': 'http://www.tei-c.org/ns/1.0'}


def find_xml_for_image(image_path: str) -> str:
    """
    Auto-detect the XML file that corresponds to the image.

    Convention: BOOKID_pdf_START-END_IMAGEINDEX_png.rf.HASH.ext
    The XML file is named:  BOOKID_pdf_START-END.xml
    and lives in the same directory as the image.
    """
    image_dir      = os.path.dirname(os.path.abspath(image_path))
    image_filename = os.path.basename(image_path)

    m = re.match(r'^(.+?_pdf_[\d-]+)_\d+_png', image_filename, re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Cannot derive XML filename from image: {image_filename}\n"
            f"Expected pattern: BOOKID_pdf_START-END_IMAGEINDEX_png..."
        )

    xml_name = m.group(1) + '.xml'
    xml_path = os.path.join(image_dir, xml_name)

    if not os.path.exists(xml_path):
        raise FileNotFoundError(
            f"Auto-detected XML file not found: {xml_path}\n"
            f"Please place it in the same folder as the image."
        )

    return xml_path


def parse_page_number(image_filename: str) -> int:
    """
    Strategy 1 (primary): extract the xml page number from the image filename.

    Filename convention: BOOKID_pdf_START-END_IMAGEINDEX_png.rf.HASH.ext
    The number before '_png' is the 1-based image index.
    The XML uses 0-based pb n=, so:  xml_page_n = image_index - 1
    """
    m = re.search(r'_(\d+)_png', image_filename, re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Could not find page number in filename: {image_filename}\n"
            f"Expected pattern: ..._<NUMBER>_png..."
        )
    image_index = int(m.group(1))
    xml_page_n  = image_index - 1
    return xml_page_n


def get_page_lines(root, page_n: int) -> list[dict]:
    """
    Return all non-empty lb tail texts for the given xml page number,
    preserving their parent tag name so callers can filter by element type.

    Each lb n="PAGE_BLOCK_LINE" stores its text as the element's tail.
    """
    prefix = f"{page_n}_"
    results = []

    for lb in root.findall('.//tei:lb', NS):
        if not lb.get('n', '').startswith(prefix):
            continue
        text = (lb.tail or '').strip()
        if not text:
            continue
        parent_tag = _find_parent_tag(root, lb)
        results.append({'text': text, 'tag': parent_tag})

    return results


def _find_parent_tag(root, target) -> str:
    """Return the local tag name of target's direct parent element."""
    for parent in root.iter():
        if target in list(parent):
            return parent.tag.split('}')[-1]
    return 'unknown'


def extract_title(lines: list[dict]) -> str | None:
    """
    Strategy 3 (fallback): find the title from <head> lines,
    excluding known non-title patterns (URLs, 'Seite N von 510').
    """
    noise = re.compile(r'^https?://|^Seite\s+\d+\s+von\s+\d+$', re.IGNORECASE)
    for item in lines:
        if item['tag'] == 'head' and not noise.match(item['text']):
            return item['text']
    return None


def extract_entry_number(lines: list[dict]) -> str | None:
    """
    Strategy 2 (fallback): find the entry number, e.g. 'N.° 83.'
    """
    pattern = re.compile(r'N[.\s°ο°]*(\d+)', re.IGNORECASE)
    for item in lines:
        m = pattern.search(item['text'])
        if m:
            return m.group(1)
    return None


def group_lines(lines: list[dict]) -> list[dict]:
    """
    Merge consecutive lines that share the same tag into a single entry.
    Each entry: { "tag": ..., "text": "combined text" }
    """
    noise = re.compile(r'^https?://|^Seite\s+\d+\s+von\s+\d+$', re.IGNORECASE)
    grouped = []
    for item in lines:
        if noise.match(item['text']):
            continue
        if grouped and grouped[-1]['tag'] == item['tag']:
            grouped[-1]['text'] += ' ' + item['text']
        else:
            grouped.append({'tag': item['tag'], 'text': item['text']})
    return grouped


def build_full_text(grouped: list[dict]) -> str:
    """
    Join all grouped entries into a single plain string, no tags.
    Each group is separated by a newline.
    """
    return '\n'.join(entry['text'] for entry in grouped)


def extract_page(xml_path: str, image_path: str) -> dict:
    """
    Main extraction function. Returns a dict with:
        image_filename - basename of the image
        image_index    - 1-based index from filename
        page_n         - xml page number (0-based)
        entry_number   - e.g. '83'  (or None)
        title          - e.g. 'DIVOZIONE, JATTANZA E MALDICENZA.' (or None)
        lines          - grouped lines: consecutive same-tag lines merged,
                         each entry is { "tag": ..., "text": ... }
        full_text      - all text joined together, no tags
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_filename = os.path.basename(image_path)
    page_n         = parse_page_number(image_filename)
    raw_lines      = get_page_lines(root, page_n)

    if not raw_lines:
        raise ValueError(
            f"No content found for xml page n={page_n}.\n"
            f"Check that '{os.path.basename(xml_path)}' matches the image."
        )

    grouped = group_lines(raw_lines)

    return {
        'image_filename': image_filename,
        'image_index':    page_n + 1,
        'page_n':         page_n,
        'entry_number':   extract_entry_number(raw_lines),
        'title':          extract_title(raw_lines),
        'lines':          grouped,
        'full_text':      build_full_text(grouped),
    }


def save_json(result: dict, image_path: str, output_path: str = None) -> str:
    """
    Save the result as a JSON file.
    - If --output is given, use that path.
    - Otherwise save in the current working directory using the image base name.
    """
    if output_path:
        out_path = output_path
    else:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path  = os.path.join(os.getcwd(), base_name + '.json')

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Extract ground truth for one book page and save as JSON.'
    )
    parser.add_argument(
        '--xml', required=True,
        help='Path to the TEI XML file'
    )
    parser.add_argument(
        '--image', required=True,
        help='Path to the page image'
    )
    parser.add_argument(
        '--output', default=None,
        help='(Optional) Path for the output JSON file. '
             'Defaults to <image_basename>.json in the current directory.'
    )
    args = parser.parse_args()

    xml_path = args.xml
    result   = extract_page(xml_path, args.image)
    out_path = save_json(result, args.image, args.output)

    print(f"{'='*60}")
    print(f"Image          : {result['image_filename']}")
    print(f"XML used       : {os.path.basename(xml_path)}")
    print(f"Image index    : {result['image_index']}")
    print(f"XML page n     : {result['page_n']}")
    print(f"Entry number   : {result['entry_number'] or '(not found)'}")
    print(f"Title          : {result['title'] or '(not found)'}")
    print(f"{'='*60}")
    print(f"JSON saved to  : {out_path}")


if __name__ == '__main__':
    main()