#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IPAT 기출문제 파일에서 서식(==표시)과 해설 부분을 삭제하는 스크립트
"""

import re
import os

def remove_format_and_explanation(input_file, output_file):
    """
    파일에서 서식(==표시)과 해설 부분을 제거합니다.
    
    Args:
        input_file (str): 입력 파일 경로
        output_file (str): 출력 파일 경로
    """
    try:
        print(f"📂 입력 파일 확인: {input_file}")
        if not os.path.exists(input_file):
            print(f"❌ 파일이 존재하지 않습니다: {input_file}")
            return
        
        print(f"📖 파일 읽기 중...")
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📊 원본 파일 크기: {len(content):,} 문자")
        
        # 1. 서식 제거: == 표시로 감싸진 부분에서 == 만 제거 (내용은 유지)
        print("🔧 서식 제거 중...")
        before_format = content
        content = re.sub(r'==([^=]+)==', r'\1', content)
        format_removed = len(before_format) - len(content)
        print(f"   → 서식 제거로 {format_removed} 문자 제거됨")
        
        # 2. 해설 부분 제거: **해설:** 부터 다음 --- 또는 **문제번호** 전까지
        print("🔧 해설 부분 제거 중...")
        before_explanation = content
        content = re.sub(r'\*\*해설:\*\*.*?(?=\n\n---|\n\*\*\d+\.|\Z)', '', content, flags=re.DOTALL)
        explanation_removed = len(before_explanation) - len(content)
        print(f"   → 해설 제거로 {explanation_removed} 문자 제거됨")
        
        # 3. 빈 줄 정리: 연속된 빈 줄을 최대 2개로 제한
        print("🔧 빈 줄 정리 중...")
        before_cleanup = content
        content = re.sub(r'\n{3,}', '\n\n', content)
        cleanup_removed = len(before_cleanup) - len(content)
        print(f"   → 빈 줄 정리로 {cleanup_removed} 문자 제거됨")
        
        # 4. 파일 끝의 불필요한 빈 줄 제거
        content = content.rstrip() + '\n'
        
        # 출력 디렉토리 생성 (필요한 경우)
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 출력 디렉토리 생성: {output_dir}")
        
        # 결과 저장
        print(f"💾 결과 저장 중...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 처리 완료!")
        print(f"📁 입력 파일: {input_file}")
        print(f"📁 출력 파일: {output_file}")
        
        # 통계 정보
        with open(input_file, 'r', encoding='utf-8') as f:
            original_lines = len(f.readlines())
        processed_lines = len(content.split('\n'))
        
        print(f"\n📊 처리 결과:")
        print(f"   • 원본 라인 수: {original_lines:,}")
        print(f"   • 처리 후 라인 수: {processed_lines:,}")
        print(f"   • 제거된 라인 수: {original_lines - processed_lines:,}")
        print(f"   • 총 제거된 문자 수: {format_removed + explanation_removed + cleanup_removed:,}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 파일 경로 설정
    input_file = r"ComputerScience\지식재산개론\기출문제\processed\IPAT_기출문제_답안해설_완료.md"
    output_file = r"ComputerScience\지식재산개론\기출문제\processed\IPAT_기출문제_서식해설제거.md"
    
    print("🚀 IPAT 기출문제 서식 및 해설 제거 시작...")
    remove_format_and_explanation(input_file, output_file) 