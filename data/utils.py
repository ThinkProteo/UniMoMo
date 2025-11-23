#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
from typing import Dict, Optional

import torch


def load_prompt_jsonl(path: str, id_key: str = 'id', prompt_key: str = 'prompt') -> Dict[str, str]:
    prompt_map: Dict[str, str] = {}
    if path is None:
        return prompt_map
    with open(path, 'r') as fin:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            if id_key not in record or prompt_key not in record:
                continue
            _id = str(record[id_key]).strip()
            _prompt = record[prompt_key]
            prompt_map[_id] = _prompt
            prompt_map[_id.lower()] = _prompt
    return prompt_map


def encode_prompt_text(prompt: Optional[str]) -> torch.Tensor:
    if prompt is None:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor([ord(c) for c in prompt], dtype=torch.long)
