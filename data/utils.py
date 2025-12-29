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


def load_prompt_jsonl_extended_dual(
    path: str,
    id_key: str = 'complex_id',
    prompt_key: str = 'question',
    thinking_key: str = 'thinking',
    response_key: str = 'answer',
    prevent_leakage_qkv_only: bool = False,
    leakage_marker: str = '**Foldability:**',
    use_answer_only_qkv: bool = False
) -> tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Load extended JSONL with TWO response versions for separate QKV and SFT processing.

    This mode allows:
    - QKV extraction: Use truncated thinking (no answer leakage)
    - SFT loss: Use full thinking + answer (supervise on complete reasoning)

    Args:
        path: Path to JSONL file
        id_key: Key for sample ID (default: 'complex_id')
        prompt_key: Key for prompt text (default: 'question')
        thinking_key: Key for CoT thinking (default: 'thinking')
        response_key: Key for response text (default: 'answer')
        prevent_leakage_qkv_only: If True, return dual responses (truncated for QKV, full for SFT)
        leakage_marker: Marker indicating start of answer content in thinking (default: '**Foldability:**')
        use_answer_only_qkv: If True, QKV only uses answer text (no thinking/foldability).
                            This is a DEBUG mode to test if diffusion can learn from ground truth answer only.

    Returns:
        (prompt_map, response_qkv_map, response_sft_map): Three dicts mapping sample_id -> text
            - prompt_map: Question text
            - response_qkv_map: Thinking truncated at marker (for QKV extraction), or answer-only if use_answer_only_qkv
            - response_sft_map: Full thinking + answer (for SFT loss)
            
        If prevent_leakage_qkv_only=False, response_qkv_map == response_sft_map (backward compatible)
    """
    prompt_map: Dict[str, str] = {}
    response_qkv_map: Dict[str, str] = {}
    response_sft_map: Dict[str, str] = {}
    raw_text_map: Dict[str, str] = {}

    if path is None:
        return prompt_map, response_qkv_map, response_sft_map

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

            # Extract thinking and answer
            _thinking = record.get(thinking_key, "")
            _answer = record.get(response_key, "")

            if prevent_leakage_qkv_only:
                # NEW MODE: Dual response versions
                raw_text_map[_id] = record
                raw_text_map[_id.lower()] = record
                
                if use_answer_only_qkv:
                    # DEBUG MODE: QKV uses only the answer text (no thinking, no foldability)
                    # Purpose: Test if diffusion can learn using ground truth answer representation
                    if _answer:
                        response_qkv_map[_id] = _answer
                        response_qkv_map[_id.lower()] = _answer
                else:
                    # 1. QKV version: Truncated thinking (no answer leakage)
                    _thinking_truncated = _thinking
                    if _thinking and leakage_marker in _thinking: #Erran: bug: the data uses "**5. Foldability:**"
                        _thinking_truncated = _thinking.split(leakage_marker)[0].strip()
                    
                    if _thinking_truncated:
                        response_qkv_map[_id] = _thinking_truncated
                        response_qkv_map[_id.lower()] = _thinking_truncated
                
                # 2. SFT version: Full thinking + answer
                response_parts = []
                if _thinking:
                    response_parts.append(_thinking)
                if _answer:
                    response_parts.append(_answer)
                
                if response_parts:
                    combined_response = "\n\n".join(response_parts)
                    response_sft_map[_id] = combined_response
                    response_sft_map[_id.lower()] = combined_response
            else:
                # BACKWARD COMPATIBLE: Same response for both QKV and SFT
                # Combine full thinking + answer
                response_parts = []
                if _thinking:
                    response_parts.append(_thinking)
                if _answer:
                    response_parts.append(_answer)
                
                if response_parts:
                    combined_response = "\n\n".join(response_parts)
                    response_qkv_map[_id] = combined_response
                    response_qkv_map[_id.lower()] = combined_response
                    response_sft_map[_id] = combined_response
                    response_sft_map[_id.lower()] = combined_response

    return prompt_map, response_qkv_map, response_sft_map, raw_text_map


def default_prompt_path(dataset_type: str) -> Optional[str]:
    """
    Return default prompt JSONL path for a given dataset type.
    
    This is a backward compatibility function for datasets that don't explicitly
    provide a prompt_jsonl path. Returns None to indicate no default prompts.
    
    Args:
        dataset_type: Type of dataset ('antibody', 'peptide', 'molecule', etc.)
    
    Returns:
        Path to default prompt file, or None if no default exists
    """
    # For inference/generation, prompts should be explicitly provided via config
    # Return None to avoid silent failures
    return None
