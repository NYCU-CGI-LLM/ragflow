#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json
import re
from rag.nlp import find_codec, rag_tokenizer
from rag.app.naive import tokenize_chunks

def chunk(filename, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=None, **kwargs):
    """
    Splits a JSON file into chunks, where each JSON object is a chunk.
    """
    is_english = lang.lower() == "english"
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
    }
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    
    sections = []
    if binary:
        encoding = find_codec(binary)
        content = binary.decode(encoding, errors="ignore")
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    sections.append(json.dumps(item, ensure_ascii=False))
            elif isinstance(data, dict):
                sections.append(json.dumps(data, ensure_ascii=False))
        except json.JSONDecodeError:
            callback(0.8, "Failed to decode JSON.")
            return []
    
    res = tokenize_chunks(sections, doc, is_english)
    return res

