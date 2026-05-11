import pdfplumber
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PDF_DIR = BASE_DIR / "data" / "raw"
OUTPUT_TEXT_DIR = BASE_DIR / "data" / "processed"

# 确保输出文件夹存在
OUTPUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path, output_txt_path):
    """提取PDF所有页面文本"""
    print(f"正在处理: {pdf_path}")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    # 保存为txt
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"已保存文本: {output_txt_path}")
    print(f"共提取 {len(text)} 个字符\n")


def process_all_pdfs():
    """处理raw文件夹下所有子文件夹中的PDF"""
    raw_path = Path(RAW_PDF_DIR)
    output_path = Path(OUTPUT_TEXT_DIR)

    # 确保输出文件夹存在
    output_path.mkdir(parents=True, exist_ok=True)

    # 递归搜索所有PDF文件
    pdf_files = list(raw_path.glob('**/*.pdf'))

    if not pdf_files:
        print(f"⚠️ raw文件夹下没有找到PDF文件！目录: {raw_path}")
        return

    print(f"找到 {len(pdf_files)} 个PDF文件，开始提取文本...\n")

    for pdf_path in pdf_files:
        # 获取相对于raw文件夹的路径
        relative_path = pdf_path.relative_to(raw_path)

        # 用下划线连接路径，全部输出到同一文件夹
        txt_name = str(relative_path.with_suffix('.txt')).replace('/', '_').replace('\\', '_')
        txt_path = output_path / txt_name

        print(f"📄 处理: {relative_path}")
        extract_text_from_pdf(str(pdf_path), str(txt_path))

    print(f"\n🎉 所有PDF文本提取完成！")
    print(f"文本文件已保存到: {OUTPUT_TEXT_DIR}")



if __name__ == "__main__":
    process_all_pdfs()
