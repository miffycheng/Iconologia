# extract_page.py

Extract ground-truth text from a single page of an ancient book, identified by its image filename. The script reads a TEI XML file (containing the full book transcription) and outputs a structured JSON for the requested page.

---

## Requirements

No third-party libraries needed — only Python 3.10+ standard library.

---

## File Naming Convention

The script relies on a consistent naming pattern shared by the image files and the XML:

```
Image:  BOOKID_pdf_START-END_IMAGEINDEX_png.rf.HASH.jpg
XML:    BOOKID_pdf_START-END.xml
```

Example:
```
Image:  1026A428_pdf_1-510_352_png.rf.e065fc1d114e613b0a8991da4101f184.jpg
XML:    1026A428_pdf_1-510.xml
```

The number before `_png` in the image filename is the **1-based image index**. The XML uses **0-based** page numbers (`pb n=`), so the mapping is:

```
xml_page_n = image_index - 1
```

For example, image index `352` → XML page `n=351`, which corresponds to `Seite 351 von 510`.

---

## Usage

```bash
python3 extract_page.py --xml <xml_path> --image <image_path> [--output <output_path>]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--xml` | Yes | Path to the TEI XML file |
| `--image` | Yes | Path to the page image |
| `--output` | No | Path for the output JSON. Defaults to `<image_basename>.json` in the current directory |

### Example

```bash
python3 extract_page.py \
    --xml ./data/1026A428_pdf_1-510.xml \
    --image ./data/test/1026A428_pdf_1-510_352_png.rf.e065fc1d114e613b0a8991da4101f184.jpg \
    --output ./image_gt.json
```

---

## Output JSON

The output JSON contains the following fields:

| Field | Type | Description |
|---|---|---|
| `image_filename` | string | Basename of the input image |
| `image_index` | int | 1-based image index from the filename |
| `page_n` | int | 0-based XML page number |
| `entry_number` | string \| null | Entry number e.g. `"83"`, parsed from `N.° 83.` markers |
| `title` | string \| null | Page title from `<head>` element, e.g. `"DIVOZIONE, JATTANZA E MALDICENZA."` |
| `lines` | list | Grouped lines — consecutive lines sharing the same XML tag are merged into one entry: `{ "tag": "p", "text": "..." }` |
| `full_text` | string | All text joined in reading order, no tags, newline-separated by group |

### Example output

```json
{
  "image_filename": "1026A428_pdf_1-510_352_png.rf.e065fc1d114e613b0a8991da4101f184.jpg",
  "image_index": 352,
  "page_n": 351,
  "entry_number": "83",
  "title": "DIVOZIONE, JATTANZA E MALDICENZA.",
  "lines": [
    { "tag": "figure", "text": "N.° 83." },
    { "tag": "fw",     "text": "165" },
    { "tag": "head",   "text": "DIVOZIONE, JATTANZA E MALDICENZA." },
    { "tag": "p",      "text": "Divozione. Donna avvenente, genuflessa..." },
    { "tag": "ab",     "text": "G G G" }
  ],
  "full_text": "N.° 83.\n165\nDIVOZIONE, JATTANZA E MALDICENZA.\nDivozione. Donna avvenente, genuflessa...\nG G G"
}
```

---

## How Page Identification Works

The script uses three strategies in order of reliability:

**Strategy 1 — Filename (primary):** Parses the 1-based image index from the filename and subtracts 1 to get the XML page number. This is the main mechanism and works for all pages.

**Strategy 2 — Entry number (fallback):** Scans all lines for a `N.° <number>` pattern inside `<figure>` elements. Useful for cross-validation but note that two consecutive pages (Italian + French) often share the same entry number.

**Strategy 3 — Title (fallback):** Finds the first `<head>` element on the page that is not a URL or a `Seite N von 510` footer. Not every page has a title (e.g. image-only or blank pages).

---

## XML Structure Notes

The TEI XML encodes each line of text as a `<lb>` element whose `n` attribute follows the pattern `PAGE_BLOCK_LINE`. The text itself is stored as the element's **tail**:

```xml
<p>
  <lb n="351_5_1"/> Divozione. Donna avvenente, genuflessa...
  <lb n="351_5_2"/> con tenerezza riguarda.
</p>
```

All lines belonging to page 351 have `n` attributes starting with `"351_"`. Noise lines (URLs, page footers) are filtered out automatically.
