"""Structured feature definitions for the 11-project dataset."""

from __future__ import annotations

NUMERIC_FEATURES = [
    "Floor_Area_m2",
    "Units",
    "Storeys",
    "Bedrooms_Total",
]

CATEGORICAL_FEATURES = [
    "Foundation_Type",
    "Roof_Form",
    "Primary_Cladding",
    "Site_Complexity",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGETS = {
    "foundation": "Foundation_Target_exGST",
    "framing": "Framing_Package_Target_exGST",
    "roofing": "Roofing_Target_exGST",
}
