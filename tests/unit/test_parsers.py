"""Unit tests for VLM response parsers.

Tests the FormParser and JSONParser classes with real workshop output.
"""

from __future__ import annotations

import pytest

from seedlearn.components.analyzers import FormParser, JSONParser


# =============================================================================
# Test Data (from workshop_pipeline/step_1 results)
# =============================================================================

SAMPLE_FORM_OUTPUT = """
A. Leaf Arrangement & Architecture  
    1. Leaf relative position (alternate / opposite / whorled): whorled (leaves arranged in a circular pattern around the stem node)  
    2. Leaf spacing (clustered / distal): distal (leaves spaced apart along the stem)  
B. Leaf Complexity  
    3. Leaf complexity (simple / compound): simple (single undivided blade)  
    4. Compound leaf type, ONLY if leaf complexity is compound (odd-pinnate / even-pinnate): N/A (leaf is simple)  
    5. Number of leaflets (integer estimate 2, 4, 8, …): N/A (leaf is simple)  
    6. Leaflet arrangement (opposite / alternate / subopposite): N/A (leaf is simple)  
C. Leaf Morphology  
    7. Leaf margin (entire / toothed) AND if toothed (dentate, serrate, etc.)): entire (smooth margins with no teeth)  
    8. Leaf shape (elliptic, obovate, lanceolate, etc.): elliptic (oval shape with widest point near the middle)  
    9. Leaf apex (acute, obtuse, acuminate): acute (sharply pointed tip)  
    10. Leaf base (rounded, cordate, cuneate): rounded (base gently curved without a notch)  
    11. Venation type (pinnate / palmate / parallel): pinnate (secondary veins branching from a central midrib)  
    12. Secondary veins (visibility, spacing, number): visible, moderately spaced (clear veins with consistent spacing)  
    13. Leaf surface features (glabrous, shiny, dull, rugose): glabrous, shiny (smooth, glossy surface without texture)  
    14. Leaf surface trichomes (present / absent): absent (no visible hairs on the leaf surface)  
    15. Petiole length (short / long): short (petioles are brief and not elongated)  
    16. Petiole features (winged / grooved / terete): terete (cylindrical, rounded petiole)  
D. Stem & Shoot Traits  
    17. Stem type, may not be visible (woody / herbaceous): woody (stems appear rigid and tree-like)  
    18. Stem trichomes (present / absent): absent (no visible hairs on the stem)  
    19. Stem color: green (young, greenish stem)  
    20. Stem texture (smooth / ridged / lenticellate): smooth (even surface without ridges or lenticels)  
E. Other Visible Seedling Traits  
    21. Stipules, sometimes visible only in close-up images (present / absent): absent (no stipules visible at leaf bases)  
    22. Latex, rarely visible but extremely diagnostic when present (present / absent): absent (no latex observed)  
    23. Pulvinus, for Fabaceae family only (present / absent): absent (not a Fabaceae species)  
    24. Tendrils (present / absent): absent (no coiling structures present)  
F. Notes: Whorled leaf arrangement with multiple leaves per node; some leaf damage (holes) likely from insects; forest floor habitat with leaf litter.  

####
"""

SAMPLE_JSON_OUTPUT = """{
  "leaf_arrangement": {
    "relative_position": "alternate",
    "spacing": "distal"
  },
  "leaf_complexity": {
    "type": "compound",
    "compound_type": "odd-pinnate",
    "num_leaflets": 7,
    "leaflet_arrangement": "opposite"
  },
  "leaf_morphology": {
    "margin": "entire",
    "shape": "elliptic",
    "apex": "acuminate",
    "base": "rounded",
    "venation": "pinnate",
    "secondary_veins": "visible, moderately spaced",
    "surface_features": "glabrous",
    "trichomes": "absent",
    "petiole_length": "short",
    "petiole_features": "terete"
  },
  "stem_traits": {
    "type": "woody",
    "trichomes": "absent",
    "color": "green-brown",
    "texture": "smooth"
  },
  "special_features": {
    "stipules": "present",
    "latex": "absent",
    "pulvinus": "present",
    "tendrils": "absent"
  },
  "notes": "Likely Fabaceae based on pulvinus and compound leaves"
}"""


# =============================================================================
# FormParser Tests
# =============================================================================

class TestFormParser:
    """Tests for FormParser."""
    
    def test_parse_basic_structure(self) -> None:
        """Test parsing creates correct structure."""
        result = FormParser.parse(SAMPLE_FORM_OUTPUT)
        
        assert "leaf_arrangement" in result
        assert "leaf_complexity" in result
        assert "leaf_morphology" in result
        assert "stem_traits" in result
        assert "special_features" in result
        assert "notes" in result
    
    def test_parse_leaf_arrangement(self) -> None:
        """Test leaf arrangement fields are parsed."""
        result = FormParser.parse(SAMPLE_FORM_OUTPUT)
        
        assert result["leaf_arrangement"]["relative_position"] == "whorled"
        assert result["leaf_arrangement"]["spacing"] == "distal"
    
    def test_parse_leaf_complexity(self) -> None:
        """Test leaf complexity fields are parsed."""
        result = FormParser.parse(SAMPLE_FORM_OUTPUT)
        
        assert result["leaf_complexity"]["type"] == "simple"
        assert result["leaf_complexity"]["compound_type"] == "N/A"
    
    def test_parse_strips_justifications(self) -> None:
        """Test that parenthetical justifications are stripped."""
        result = FormParser.parse(SAMPLE_FORM_OUTPUT)
        
        # Should be "whorled" not "whorled (leaves arranged...)"
        assert "(" not in result["leaf_arrangement"]["relative_position"]
        assert result["leaf_arrangement"]["relative_position"] == "whorled"
    
    def test_parse_notes(self) -> None:
        """Test notes field is parsed."""
        result = FormParser.parse(SAMPLE_FORM_OUTPUT)
        
        assert "Whorled leaf arrangement" in result["notes"]
    
    def test_to_morphology_result(self) -> None:
        """Test conversion to MorphologyResult."""
        parsed = FormParser.parse(SAMPLE_FORM_OUTPUT)
        result = FormParser.to_morphology_result(parsed, image_id="test_001", time_ms=150.0)
        
        assert result.image_id == "test_001"
        assert result.processing_time_ms == 150.0
        assert result.leaf_arrangement.relative_position == "whorled"
        assert result.leaf_complexity.type == "simple"


# =============================================================================
# JSONParser Tests
# =============================================================================

class TestJSONParser:
    """Tests for JSONParser."""
    
    def test_parse_valid_json(self) -> None:
        """Test parsing valid JSON."""
        result = JSONParser.parse(SAMPLE_JSON_OUTPUT)
        
        assert result["leaf_complexity"]["type"] == "compound"
        assert result["leaf_complexity"]["num_leaflets"] == 7
    
    def test_parse_with_markdown(self) -> None:
        """Test parsing JSON wrapped in markdown."""
        markdown_json = f"```json\n{SAMPLE_JSON_OUTPUT}\n```"
        result = JSONParser.parse(markdown_json)
        
        assert result["leaf_complexity"]["type"] == "compound"
    
    def test_to_morphology_result(self) -> None:
        """Test conversion to MorphologyResult."""
        parsed = JSONParser.parse(SAMPLE_JSON_OUTPUT)
        result = JSONParser.to_morphology_result(parsed, image_id="test_002", time_ms=100.0)
        
        assert result.image_id == "test_002"
        assert result.leaf_complexity.type == "compound"
        assert result.leaf_complexity.num_leaflets == 7
        assert result.special_features.pulvinus == "present"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
