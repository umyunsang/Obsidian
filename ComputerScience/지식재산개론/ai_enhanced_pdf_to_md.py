import os
import sys
import fitz  # PyMuPDF
from openai import OpenAI
import json
import time

class AIEnhancedPDFConverter:
    def __init__(self, api_key=None):
        """AI 기반 PDF to Markdown 변환기 초기화"""
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # 환경변수에서 API 키를 읽어옴
            self.client = OpenAI()
        
        self.image_dir = "pdf_images"
        
    def extract_text_from_pdf(self, pdf_path):
        """PDF에서 텍스트 추출"""
        doc = fitz.open(pdf_path)
        pages_text = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            pages_text.append({
                'page_num': page_num + 1,
                'text': text.strip()
            })
        
        doc.close()
        return pages_text
    
    def save_page_images(self, pdf_path):
        """PDF 페이지를 이미지로 저장"""
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        
        doc = fitz.open(pdf_path)
        image_paths = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            matrix = fitz.Matrix(2.0, 2.0)  # 고해상도
            pix = page.get_pixmap(matrix=matrix)
            
            image_filename = f"page_{page_num + 1:02d}.png"
            image_path = os.path.join(self.image_dir, image_filename)
            pix.save(image_path)
            image_paths.append(image_path)
            
            print(f"페이지 {page_num + 1} 이미지 저장됨: {image_path}")
        
        doc.close()
        return image_paths
    
    def enhance_text_with_ai(self, text, page_num, context="특허 명세서 강의자료"):
        """AI를 사용해서 텍스트를 마크다운으로 개선"""
        if not text.strip():
            return f"*[페이지 {page_num}에 텍스트 내용 없음]*"
        
        prompt = f"""
다음은 {context}의 페이지 {page_num} 내용입니다. 이 텍스트를 보기 좋은 마크다운 형식으로 정리해주세요.

요구사항:
1. 내용을 그대로 유지하되, 마크다운 서식을 적절히 적용
2. 제목은 적절한 헤딩 레벨(#, ##, ###) 사용
3. 중요한 내용은 **볼드** 처리
4. 목록이 있으면 - 또는 1. 형식으로 정리
5. 코드나 특별한 용어는 `백틱`으로 감싸기
6. 긴 문장은 읽기 쉽게 줄바꿈 적용
7. 한국어 맞춤법과 띄어쓰기 교정
8. 표가 있으면 마크다운 표 형식으로 변환
9. 강조할 내용은 > 인용구 사용

원본 텍스트:
{text}

마크다운으로 정리된 결과만 출력해주세요:
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 문서 정리 전문가입니다. 주어진 텍스트를 보기 좋은 마크다운 형식으로 정리해주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            enhanced_text = response.choices[0].message.content.strip()
            print(f"페이지 {page_num} AI 처리 완료")
            
            # API 호출 제한을 위한 잠시 대기
            time.sleep(0.5)
            
            return enhanced_text
            
        except Exception as e:
            print(f"AI 처리 중 오류 발생 (페이지 {page_num}): {e}")
            return f"### 페이지 {page_num} 원본 텍스트\n\n{text}"
    
    def create_enhanced_markdown(self, pdf_path, pages_text, image_paths, output_path):
        """개선된 마크다운 파일 생성"""
        filename = os.path.basename(pdf_path).replace('.pdf', '')
        
        # 제목 및 메타데이터
        markdown_content = f"# {filename}\n\n"
        markdown_content += f"> **원본 파일**: {pdf_path}\n"
        markdown_content += f"> **총 페이지 수**: {len(pages_text)}\n"
        markdown_content += f"> **변환 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        markdown_content += f"> **처리 방식**: AI 기반 텍스트 개선 + 이미지 변환\n\n"
        
        # 목차 생성을 위한 AI 호출
        all_text = "\n\n".join([page['text'] for page in pages_text if page['text']])
        toc_prompt = f"""
다음 문서의 내용을 분석해서 목차를 만들어주세요. 마크다운 링크 형식으로 작성해주세요.

문서 내용:
{all_text[:3000]}...

목차 형식:
- [주요 섹션명](#섹션명)
  - [하위 섹션명](#하위-섹션명)

목차만 출력해주세요:
"""
        
        try:
            toc_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "문서 분석 및 목차 생성 전문가입니다."},
                    {"role": "user", "content": toc_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            toc = toc_response.choices[0].message.content.strip()
            markdown_content += f"## 📋 목차\n\n{toc}\n\n"
            
        except Exception as e:
            print(f"목차 생성 중 오류: {e}")
        
        markdown_content += "---\n\n"
        
        # 각 페이지 처리
        for i, page_data in enumerate(pages_text):
            page_num = page_data['page_num']
            text = page_data['text']
            
            print(f"페이지 {page_num} AI 처리 중...")
            
            # AI로 텍스트 개선
            enhanced_text = self.enhance_text_with_ai(text, page_num)
            
            # 페이지 섹션 추가
            markdown_content += f"## 📄 페이지 {page_num}\n\n"
            markdown_content += f"{enhanced_text}\n\n"
            
            # 이미지 참조 추가
            if i < len(image_paths):
                markdown_content += f"### 🖼️ 페이지 이미지\n\n"
                markdown_content += f"![페이지 {page_num}]({image_paths[i]})\n\n"
            
            markdown_content += "---\n\n"
        
        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"개선된 마크다운 파일 저장됨: {output_path}")
    
    def convert_pdf_to_enhanced_markdown(self, pdf_path, output_path=None):
        """PDF를 AI 기반으로 개선된 마크다운으로 변환"""
        if not os.path.exists(pdf_path):
            print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            return False
        
        if output_path is None:
            output_path = pdf_path.replace('.pdf', '_ai_enhanced.md')
        
        print(f"🤖 AI 기반 PDF to Markdown 변환 시작: {pdf_path}")
        
        try:
            # 1. 텍스트 추출
            print("📝 텍스트 추출 중...")
            pages_text = self.extract_text_from_pdf(pdf_path)
            
            # 2. 이미지 저장
            print("🖼️ 페이지 이미지 저장 중...")
            image_paths = self.save_page_images(pdf_path)
            
            # 3. AI로 마크다운 생성
            print("🤖 AI를 사용한 텍스트 개선 중...")
            self.create_enhanced_markdown(pdf_path, pages_text, image_paths, output_path)
            
            print(f"✅ 변환 완료! 결과 파일: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ 변환 중 오류 발생: {e}")
            return False

def main():
    if len(sys.argv) < 2:
        print("사용법: python ai_enhanced_pdf_to_md.py <PDF파일경로> [출력MD파일경로] [OpenAI_API_키]")
        print("참고: OpenAI API 키는 환경변수 OPENAI_API_KEY로도 설정 가능")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    api_key = sys.argv[3] if len(sys.argv) > 3 else None
    
    # API 키 확인
    if not api_key and not os.getenv('OPENAI_API_KEY'):
        print("⚠️ OpenAI API 키가 필요합니다.")
        print("1. 환경변수 OPENAI_API_KEY 설정")
        print("2. 또는 명령행 인수로 API 키 제공")
        sys.exit(1)
    
    converter = AIEnhancedPDFConverter(api_key)
    success = converter.convert_pdf_to_enhanced_markdown(pdf_file, output_file)
    
    if success:
        print("🎉 AI 기반 PDF to Markdown 변환이 완료되었습니다!")
    else:
        print("💥 변환에 실패했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main() 