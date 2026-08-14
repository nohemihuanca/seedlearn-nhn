"""Tests for seedlearn.benchmarking.id_grader."""

from seedlearn.benchmarking.id_grader import IDGradeRecord, grade_specimen_id


def _make_pipeline_result(
    family: str | None = "Fabaceae",
    genus: str | None = "Inga",
    species: str | None = "Inga marginata",
    confidence: str | None = "high",
    error: str | None = None,
    skipped: bool = False,
) -> dict:
    """Build a minimal pipeline result dict."""
    reasoning: dict = {
        "data": {
            "classification": {
                "predicted_family": family,
                "predicted_genus": genus,
                "predicted_species": species,
                "confidence": confidence,
            }
        },
        "skipped": skipped,
        "error": error,
        "elapsed_ms": 100.0,
    }
    return {
        "specimen_id": "SPEC1",
        "stages": {"reasoning": reasoning},
    }


class TestGradeSpecimenId:
    """Test grade_specimen_id function."""

    def test_all_correct(self):
        result = _make_pipeline_result()
        record = grade_specimen_id(
            "SPEC1", "Fabaceae", "Inga", "Inga marginata", result
        )

        assert record.family_correct is True
        assert record.genus_correct is True
        assert record.species_correct is True
        assert record.stage5_error is False

    def test_case_insensitive(self):
        result = _make_pipeline_result(
            family="fabaceae", genus="inga", species="inga marginata"
        )
        record = grade_specimen_id(
            "SPEC1", "Fabaceae", "Inga", "Inga marginata", result
        )

        assert record.family_correct is True
        assert record.genus_correct is True
        assert record.species_correct is True

    def test_epithet_only_matches_binomial(self):
        """Stage 5 returns just 'marginata', true is 'Inga marginata'."""
        result = _make_pipeline_result(species="marginata")
        record = grade_specimen_id(
            "SPEC1", "Fabaceae", "Inga", "Inga marginata", result
        )
        assert record.species_correct is True

    def test_epithet_only_wrong(self):
        """Stage 5 returns just 'edulis', true is 'Inga marginata'."""
        result = _make_pipeline_result(species="edulis")
        record = grade_specimen_id(
            "SPEC1", "Fabaceae", "Inga", "Inga marginata", result
        )
        assert record.species_correct is False

    def test_wrong_species(self):
        result = _make_pipeline_result(species="Inga edulis")
        record = grade_specimen_id(
            "SPEC1", "Fabaceae", "Inga", "Inga marginata", result
        )

        assert record.family_correct is True
        assert record.genus_correct is True
        assert record.species_correct is False

    def test_wrong_family(self):
        result = _make_pipeline_result(family="Rubiaceae")
        record = grade_specimen_id(
            "SPEC1", "Fabaceae", "Inga", "Inga marginata", result
        )

        assert record.family_correct is False
        assert record.genus_correct is True

    def test_stage5_error(self):
        result = _make_pipeline_result(error="VLM timeout")
        record = grade_specimen_id(
            "SPEC1", "Fabaceae", "Inga", "Inga marginata", result
        )

        assert record.stage5_error is True
        assert record.family_correct is False
        assert record.genus_correct is False
        assert record.species_correct is False

    def test_stage5_skipped(self):
        result = _make_pipeline_result(skipped=True)
        record = grade_specimen_id(
            "SPEC1", "Fabaceae", "Inga", "Inga marginata", result
        )

        assert record.stage5_error is True

    def test_null_predictions(self):
        result = _make_pipeline_result(
            family=None, genus=None, species=None
        )
        record = grade_specimen_id(
            "SPEC1", "Fabaceae", "Inga", "Inga marginata", result
        )

        assert record.family_correct is False
        assert record.genus_correct is False
        assert record.species_correct is False
        assert record.stage5_error is False

    def test_missing_reasoning_stage(self):
        result = {"specimen_id": "SPEC1", "stages": {}}
        record = grade_specimen_id(
            "SPEC1", "Fabaceae", "Inga", "Inga marginata", result
        )

        # Missing stage treated as error
        assert record.stage5_error is False  # not an error, just empty
        assert record.family_correct is False

    def test_confidence_preserved(self):
        result = _make_pipeline_result(confidence="medium")
        record = grade_specimen_id(
            "SPEC1", "Fabaceae", "Inga", "Inga marginata", result
        )

        assert record.confidence == "medium"

    def test_partition_defaults_to_none(self):
        result = _make_pipeline_result()
        record = grade_specimen_id(
            "SPEC1", "Fabaceae", "Inga", "Inga marginata", result
        )
        assert record.partition is None

    def test_partition_set_on_normal_record(self):
        result = _make_pipeline_result()
        record = grade_specimen_id(
            "SPEC1", "Fabaceae", "Inga", "Inga marginata", result,
            partition="test",
        )
        assert record.partition == "test"

    def test_partition_set_on_error_record(self):
        result = _make_pipeline_result(error="VLM timeout")
        record = grade_specimen_id(
            "SPEC1", "Fabaceae", "Inga", "Inga marginata", result,
            partition="train",
        )
        assert record.partition == "train"
        assert record.stage5_error is True
