"""
Pydantic 스키마 경계값 테스트
"""
import pytest
from pydantic import ValidationError
from src.api.schemas import PatientInput


VALID = {
    "has_asthma": 0, "has_atopic_derm": 0,
    "has_food_allergy": 0, "food_allergy_count": 0,
    "rhinitis_onset_age": 8.0, "rhinitis_duration": 2.0,
    "atopic_march": 0,
}


class TestPatientInput:
    def test_valid_input(self):
        p = PatientInput(**VALID)
        assert p.has_asthma == 0

    @pytest.mark.parametrize("field,value", [
        ("has_asthma",       2),
        ("has_asthma",      -1),
        ("has_atopic_derm",  2),
        ("has_food_allergy", 2),
        ("atopic_march",     2),
        ("food_allergy_count", 11),
        ("food_allergy_count", -1),
        ("rhinitis_onset_age", -0.1),
        ("rhinitis_onset_age", 20.1),
        ("rhinitis_duration",  -1.0),
    ])
    def test_out_of_range_raises(self, field, value):
        data = {**VALID, field: value}
        with pytest.raises(ValidationError):
            PatientInput(**data)

    @pytest.mark.parametrize("field", ["has_asthma", "has_atopic_derm",
                                        "has_food_allergy", "rhinitis_onset_age"])
    def test_missing_required_field_raises(self, field):
        data = {k: v for k, v in VALID.items() if k != field}
        with pytest.raises(ValidationError):
            PatientInput(**data)

    def test_optional_fields_have_defaults(self):
        required_only = {
            "has_asthma": 0, "has_atopic_derm": 0,
            "has_food_allergy": 0, "rhinitis_onset_age": 7.0,
        }
        p = PatientInput(**required_only)
        assert p.food_allergy_count == 0
        assert p.rhinitis_duration == 0.0
        assert p.atopic_march == 0
