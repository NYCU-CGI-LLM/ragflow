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

def _json_obj_to_text(obj):
    """
    Converts a JSON object (dict) to a key-value string format.
    e.g., {"name": "A", "def": "B"} -> "name: A\ndef: B"
    """
    if not isinstance(obj, dict):
        return str(obj)

    text_parts = []
    for key, value in obj.items():
        s_val = str(value).strip()
        if s_val:
            text_parts.append(f"{key}: {s_val}")
    return "\n".join(text_parts)


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
                    sections.append(_json_obj_to_text(item))
            elif isinstance(data, dict):
                sections.append(_json_obj_to_text(data))
        except json.JSONDecodeError:
            callback(0.8, "Failed to decode JSON.")
            return []
    
    res = tokenize_chunks(sections, doc, is_english)
    return res

