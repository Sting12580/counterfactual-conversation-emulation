from cce_data.text import extract_action_from_note, merge_speaker_intervals, normalize_header


def test_normalize_header_alias():
    assert normalize_header("GENHX") == "history of present illness"
    assert normalize_header("A/P") == "a/p"


def test_extract_action_from_note_sections():
    note = """
    Subjective: cough for 3 days.
    Assessment: likely viral URI.
    Plan: fluids, return precautions.
    """
    action, method, reasons = extract_action_from_note(note)
    assert "likely viral URI" in action
    assert "return precautions" in action
    assert method == "section_labels"
    assert reasons == []


def test_merge_textgrid_intervals():
    doctor = 'xmin = 0\nxmax = 1\ntext = "Hello"\n'
    patient = 'xmin = 0.5\nxmax = 1.5\ntext = "Hi"\n'
    merged = merge_speaker_intervals(doctor, patient)
    assert merged.splitlines() == ["Doctor: Hello", "Patient: Hi"]
