import re
from typing import Optional, List
# from vinorm import TTSnorm
from pyvinorm import ViNormalizer
import re
from difflib import SequenceMatcher
def load_mapping(mapping_file: str) -> dict:
    mapping = {}
    with open(mapping_file, "r", encoding="utf-8") as f:
        for line in f:
            if "," in line:
                left, right = line.strip().split(",", 1)
                left = left.strip().lower()
                right = right.strip().lower()
                mapping[left] = right   
    return mapping

import re
normalizer = ViNormalizer(downcase=True)
viet_pronounce = {
    'A': ', a', 'B': 'bê', 'C': 'xê', 'D': 'dê', 'E': ', e',
    'F': 'ép', 'G': 'gờ', 'H': 'hát', 'I': ', i', 'J': 'di',
    'K': 'ca', 'L': 'lờ', 'M': 'mờ', 'N': 'nờ', 'O': ', o',
    'P': 'bê', 'Q': 'quy', 'R': 'rờ', 'S': 'ét', 'T': 'tê',
    'U': ', u', 'V': 'vê', 'W': 'vê kép', 'X': 'ích', 'Y': ', y', 'Z': 'dét'
}

def pronounce_abbr(word):
    return ' '.join([viet_pronounce.get(c.upper(), c) for c in word])

def replace_abbr(text):
    return re.sub(r'\b[A-Z]{2,4}\b', lambda m: pronounce_abbr(m.group()), text)

def normalize_text_mapping(text: str, mapping: dict) -> str:
    for correct in sorted(mapping.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(correct) + r"\b"
        text = re.sub(pattern, mapping[correct], text)
    return text

# def normalize_text(text: str) -> str:
#     norm = text.lower()
#     mapping = load_mapping("filtered.txt")

#     norm = normalize_text_mapping(norm, mapping)
#     norm =replace_abbr(norm)
#     norm = norm.replace("*", "")
#     norm = TTSnorm(norm)
#     norm = re.sub(r'\s+', ' ', norm)
#     norm = re.sub(r'\s+([,\.!?;:])', r'\1', norm)
#     norm = norm.replace("#", "").replace("..",".").replace(".,",",").replace(",.",".")
#     norm = re.sub(r'([\.!?])\1+', r'\1', norm)
#     # if not re.search(r'[\.!?]$', norm):
#     #     norm += '.'
#     norm = norm.strip()
#     return norm
def is_roman_numeral(text: str) -> bool:
    """Kiểm tra xem có phải số La Mã không"""
    roman_pattern = r'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(roman_pattern, text.upper()))
def normalize_text(text: str) -> str:
    """
    Giải pháp hybrid: Kết hợp cả hai approach
    """
    import re
    
    # Step 1: Xử lý abbreviations với ưu tiên filtered.txt
    norm = text.lower()
    mapping = load_mapping("filtered.txt")
    
    # Tìm abbreviations
    abbr_pattern = r'\b[A-Z]{2,}\b'
    abbreviations = re.findall(abbr_pattern, text)
    
    # Xử lý từng abbreviation
    for abbr in abbreviations:
        if is_roman_numeral(abbr):
            continue
        abbr_lower = abbr.lower()
        
        if abbr_lower in mapping:
            # Dùng mapping từ filtered.txt
            norm = norm.replace(abbr_lower, mapping[abbr_lower])
        else:
            # Dùng replace_abbr function
            try:
                expanded = replace_abbr(abbr)
                norm = norm.replace(abbr_lower, expanded.lower())
            except:
                # Nếu replace_abbr fail, giữ nguyên
                pass
    
    # Step 2: Xử lý mapping cho phần còn lại
    norm = normalize_text_mapping(norm, mapping)
    
    # Step 3: Các bước xử lý khác
    norm = norm.replace("*", "")
   # norm = TTSnorm(norm)
    norm =normalizer.normalize(norm)
    norm = re.sub(r'\s+', ' ', norm)
    norm = re.sub(r'\s+([,\.!?;:])', r'\1', norm)
    norm = norm.replace("#", "").replace("..",".").replace(".,",",").replace(",.",".")
    norm = re.sub(r'([\.!?])\1+', r'\1', norm)
    norm = norm.strip()
    return norm
import re
from typing import List

def split_text_into_chunks(text: str) -> List[str]:
    
    if not text or not text.strip():
        return []

    sentence_pattern = re.compile(r'[^.!?]*[.!?]|[^.!?]+$', re.UNICODE)

    sentences = [s.strip() for s in sentence_pattern.findall(text) if s.strip()]

    return sentences
# def split_text_into_chunks(text: str, min_words: int = 12, max_words: int = 20) -> List[str]:
#     """
#     Chia văn bản thành các đoạn (chunk) theo câu, không cắt một câu ở giữa.
#     """
#     if not text or not text.strip():
#         return []

#     sentence_pattern = re.compile(r'(?<=\S[\.!?…])\s+|(?<=\n)+')
#     raw_sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]

#     if not raw_sentences:
#         raw_sentences = [s.strip() for s in text.splitlines() if s.strip()]

#     chunks: List[str] = []
#     current_chunk_sentences: List[str] = []
#     current_word_count = 0

#     def flush_current_chunk():
#         nonlocal current_chunk_sentences, current_word_count
#         if current_chunk_sentences:
#             chunks.append(' '.join(current_chunk_sentences).strip())
#         current_chunk_sentences = []
#         current_word_count = 0

#     for s in raw_sentences:
#         words_in_sentence = len(s.split())

#         if current_word_count == 0:
#             if words_in_sentence <= max_words:
#                 current_chunk_sentences.append(s)
#                 current_word_count = words_in_sentence
#             else:
#                 chunks.append(s)
#                 current_chunk_sentences = []
#                 current_word_count = 0
#         else:
#             if current_word_count + words_in_sentence <= max_words:
#                 current_chunk_sentences.append(s)
#                 current_word_count += words_in_sentence
#             else:
#                 flush_current_chunk()
#                 if words_in_sentence <= max_words:
#                     current_chunk_sentences.append(s)
#                     current_word_count = words_in_sentence
#                 else:
#                     chunks.append(s)
#                     current_chunk_sentences = []
#                     current_word_count = 0

#     flush_current_chunk()

#     if len(chunks) >= 2:
#         last_words = len(chunks[-1].split())
#         prev_words = len(chunks[-2].split())
#         if last_words < min_words and prev_words + last_words <= max_words:
#             chunks[-2] = (chunks[-2] + ' ' + chunks[-1]).strip()
#             chunks.pop()
#     print('-----------------',chunks)
#     return chunks
# if __name__ == "__main__":
#     mapping = load_mapping("/home/data/CUONG/ZipVoice/filtered.txt")
#     text="""Hôm nay tôi gửi email cho anh chị.
# Tối qua tôi gọi alo bằng viettel.
# Tôi đang dùng wifi để vào internet."""

#     normalized_text = normalize_text_mapping(text, mapping)
#     print(normalized_text)
  