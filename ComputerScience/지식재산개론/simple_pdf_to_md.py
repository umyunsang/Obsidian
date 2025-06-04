import os
import sys
from pdf2image import convert_from_path
from PIL import Image

def pdf_to_images_and_markdown(pdf_path, output_md_path=None):
    """PDF를 이미지로 변환하고 마크다운 생성"""
    if not os.path.exists(pdf_path):
        print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return False
    
    if output_md_path is None:
        output_md_path = pdf_path.replace('.pdf', '_converted.md')
    
    # 이미지 저장 디렉토리 생성
    image_dir = "pdf_images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    print(f"PDF를 이미지로 변환 시작: {pdf_path}")
    
    try:
        # PDF를 이미지로 변환
        images = convert_from_path(pdf_path, dpi=200)
        
        # 마크다운 내용 시작
        markdown_content = f"# {os.path.basename(pdf_path).replace('.pdf', '')}\n\n"
        markdown_content += f"> 원본 파일: {pdf_path}\n"
        markdown_content += f"> 총 페이지 수: {len(images)}\n"
        markdown_content += f"> 변환된 이미지: {image_dir} 폴더 참조\n\n"
        
        for i, image in enumerate(images):
            page_num = i + 1
            image_filename = f"page_{page_num:02d}.png"
            image_path = os.path.join(image_dir, image_filename)
            
            # 이미지 저장
            image.save(image_path, "PNG")
            print(f"페이지 {page_num} 저장됨: {image_path}")
            
            # 마크다운에 이미지 참조 추가
            markdown_content += f"## 페이지 {page_num}\n\n"
            markdown_content += f"![페이지 {page_num}]({image_path})\n\n"
            markdown_content += "---\n\n"
        
        # 마크다운 파일 저장
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"마크다운 파일 저장됨: {output_md_path}")
        print(f"이미지 파일들이 {image_dir} 폴더에 저장되었습니다.")
        
        return True
        
    except Exception as e:
        print(f"변환 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python simple_pdf_to_md.py <PDF파일경로> [출력MD파일경로]")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = pdf_to_images_and_markdown(pdf_file, output_file)
    
    if success:
        print("PDF to Markdown 변환이 완료되었습니다!")
    else:
        print("변환에 실패했습니다.")
        sys.exit(1) 