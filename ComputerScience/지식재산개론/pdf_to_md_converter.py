import os
import sys
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import re

def pdf_to_images(pdf_path, output_dir="temp_images"):
    """PDF를 이미지로 변환"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"PDF 파일을 이미지로 변환 중: {pdf_path}")
    try:
        # PDF를 이미지로 변환 (해상도 높게 설정)
        images = convert_from_path(pdf_path, dpi=300)
        image_paths = []
        
        for i, image in enumerate(images):
            image_path = os.path.join(output_dir, f"page_{i+1}.png")
            image.save(image_path, "PNG")
            image_paths.append(image_path)
            print(f"페이지 {i+1} 저장됨: {image_path}")
        
        return image_paths
    except Exception as e:
        print(f"PDF 변환 중 오류 발생: {e}")
        return []

def extract_text_from_image(image_path):
    """이미지에서 텍스트 추출 (OCR)"""
    try:
        # 한국어와 영어 모두 인식하도록 설정
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang='kor+eng')
        return text
    except Exception as e:
        print(f"텍스트 추출 중 오류 발생 ({image_path}): {e}")
        return f"[이미지에서 텍스트를 추출할 수 없음: {image_path}]"

def clean_text(text):
    """텍스트 정리"""
    # 불필요한 공백 제거
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def convert_pdf_to_markdown(pdf_path, output_md_path=None):
    """PDF를 마크다운으로 변환"""
    if not os.path.exists(pdf_path):
        print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return False
    
    if output_md_path is None:
        output_md_path = pdf_path.replace('.pdf', '.md')
    
    print(f"PDF를 마크다운으로 변환 시작: {pdf_path}")
    
    # 1. PDF를 이미지로 변환
    image_paths = pdf_to_images(pdf_path)
    
    if not image_paths:
        print("이미지 변환에 실패했습니다.")
        return False
    
    # 2. 각 이미지에서 텍스트 추출
    markdown_content = f"# {os.path.basename(pdf_path).replace('.pdf', '')}\n\n"
    markdown_content += f"> 원본 파일: {pdf_path}\n"
    markdown_content += f"> 변환 일시: {os.path.basename(__file__)}\n\n"
    
    for i, image_path in enumerate(image_paths):
        print(f"페이지 {i+1} 텍스트 추출 중...")
        text = extract_text_from_image(image_path)
        cleaned_text = clean_text(text)
        
        markdown_content += f"## 페이지 {i+1}\n\n"
        if cleaned_text:
            markdown_content += f"{cleaned_text}\n\n"
        else:
            markdown_content += f"[페이지 {i+1}에서 텍스트를 추출할 수 없음]\n\n"
        
        markdown_content += "---\n\n"
    
    # 3. 마크다운 파일로 저장
    try:
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"마크다운 파일 저장됨: {output_md_path}")
        
        # 임시 이미지 파일들 정리
        import shutil
        if os.path.exists("temp_images"):
            shutil.rmtree("temp_images")
            print("임시 이미지 파일들을 정리했습니다.")
        
        return True
    except Exception as e:
        print(f"마크다운 파일 저장 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    # 사용법: python pdf_to_md_converter.py [PDF파일경로] [출력MD파일경로(선택)]
    if len(sys.argv) < 2:
        print("사용법: python pdf_to_md_converter.py <PDF파일경로> [출력MD파일경로]")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_pdf_to_markdown(pdf_file, output_file)
    
    if success:
        print("PDF to Markdown 변환이 완료되었습니다!")
    else:
        print("변환에 실패했습니다.")
        sys.exit(1) 