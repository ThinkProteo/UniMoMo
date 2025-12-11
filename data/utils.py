#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
from typing import Dict, Optional

import torch


def load_prompt_jsonl(path: str, id_key: str = 'id', prompt_key: str = 'prompt') -> Dict[str, str]:
    """Legacy function for backward compatibility. Loads only prompt text."""
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


def load_prompt_jsonl_extended(
    path: str,
    id_key: str = 'complex_id',
    prompt_key: str = 'question',
    thinking_key: str = 'thinking',
    response_key: str = 'answer',
    prevent_leakage: bool = True,
    leakage_marker: str = '**Foldability:**'
) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Load extended JSONL with separate prompt and response fields.

    Args:
        path: Path to JSONL file
        id_key: Key for sample ID (default: 'complex_id')
        prompt_key: Key for prompt text (default: 'question')
        thinking_key: Key for CoT thinking (default: 'thinking')
        response_key: Key for response text (default: 'answer')
        prevent_leakage: If True, truncate thinking at leakage_marker and ignore response_key (default: True)
        leakage_marker: Marker indicating start of answer content in thinking (default: '**Foldability:**')

    Returns:
        (prompt_map, response_map): Two dicts mapping sample_id -> text
                                    If prevent_leakage=True: response_map contains thinking truncated at marker
                                    If prevent_leakage=False: response_map contains thinking + answer combined
    """
    prompt_map: Dict[str, str] = {}
    response_map: Dict[str, str] = {}

    if path is None:
        return prompt_map, response_map

    with open(path, 'r') as fin:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)

            # ID is required
            if id_key not in record:
                continue

            _id = str(record[id_key]).strip()

            # Extract prompt (question)
            _prompt = record.get(prompt_key, "")
            if _prompt:
                prompt_map[_id] = _prompt
                prompt_map[_id.lower()] = _prompt

            # Extract thinking
            _thinking = record.get(thinking_key, "")

            if prevent_leakage:
                # Option 1: Prevent data leakage
                # Truncate thinking at leakage marker and ignore response key
                if _thinking and leakage_marker in _thinking:
                    # Keep only content before the marker
                    _thinking = _thinking.split(leakage_marker)[0].strip()

                # Use only the truncated thinking as response (ignore response_key)
                if _thinking:
                    response_map[_id] = _thinking
                    response_map[_id.lower()] = _thinking
            else:
                # Original behavior: Combine thinking + answer
                _response = record.get(response_key, "")

                response_parts = []
                if _thinking:
                    response_parts.append(_thinking)
                if _response:
                    response_parts.append(_response)

                if response_parts:
                    combined_response = "\n\n".join(response_parts)
                    response_map[_id] = combined_response
                    response_map[_id.lower()] = combined_response

    return prompt_map, response_map


def encode_prompt_text(prompt: Optional[str]) -> torch.Tensor:
    if prompt is None:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor([ord(c) for c in prompt], dtype=torch.long)
