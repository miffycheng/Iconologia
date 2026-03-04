import os
import fitz  # PyMuPDF

def pdf_to_jpg(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    prefix: str = "page"
):
    """
    Convert a PDF into JPG images (one per page).

    Args:
        pdf_path (str): Path to input PDF
        output_dir (str): Directory to save JPG files
        dpi (int): Image resolution (default: 300)
        prefix (str): Output filename prefix

    Returns:
        List[str]: Paths of generated JPG files
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    output_paths = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        output_path = os.path.join(output_dir, f"{prefix}_{i+1}.jpg")
        pix.save(output_path)
        output_paths.append(output_path)

    doc.close()
    return output_paths

images = pdf_to_jpg(
    pdf_path="./data/Amsterdam 1698 (Jean Baudoin)/raw/Amsterdam 1698 (Jean Baudoin).pdf",
    output_dir="./data/Amsterdam 1698 (Jean Baudoin)/JPG",
    dpi=300
)
