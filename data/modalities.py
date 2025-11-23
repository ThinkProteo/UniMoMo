#!/usr/bin/python
# -*- coding:utf-8 -*-

MODALITY_TO_ID = {
    'peptide': 0,
    'molecule': 1,
    'antibody': 2,
}

ID_TO_MODALITY = {idx: name for name, idx in MODALITY_TO_ID.items()}
