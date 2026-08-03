import json

from src.evaluation_v2.quick import prepare_quick, quick_adapter


def test_prepare_quick_balances_covered_verses(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("\n".join(json.dumps({"chunk_type": "verse", "verse_ref": f"BhG 1.{i}"}) for i in (1, 2)), encoding="utf-8")
    source = tmp_path / "clean_source.jsonl"
    rows = []
    for index, verse in enumerate((1, 2)):
        rows.extend([
            {"id": f"hi-{index}", "question": f"hindi {verse}", "answer": "a", "chapter_no": 1, "verse_no": verse, "language": "hindi"},
            {"id": f"gu-{index}", "question": f"gujarati {verse}", "answer": "a", "chapter_no": 1, "verse_no": verse, "language": "gujarati"},
        ])
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    config = {"seed": 7, "paths": {"evaluation_root": str(tmp_path / "evaluation"), "chunks": str(chunks)}, "datasets": {"bhagavad_gita_qa": {"path": str(source), "version": "test"}}, "split": {"seed": 7}}
    manifest = prepare_quick(config)
    assert manifest["tracks"]["without_id_gita_qa"]["selected_examples"] == 2
    assert manifest["tracks"]["without_id_gita_qa"]["questions_per_covered_verse"] == [1]
    assert manifest["tracks"]["with_id"]["selected_examples"] == 2
    assert len(quick_adapter(config, "quick_bhagavad_gita_qa").load()) == 2


def test_clean_quick_selection_prefers_non_template_question(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps({"chunk_type": "verse", "verse_ref": "BhG 1.1"}), encoding="utf-8")
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in [
        {"id": "bad", "question": "What does this teaching on This mean?", "answer": "a", "chapter_no": 1, "verse_no": 1, "language": "english"},
        {"id": "good", "question": "What does the verse teach about the battlefield?", "answer": "a", "chapter_no": 1, "verse_no": 1, "language": "english"},
    ]), encoding="utf-8")
    config = {"seed": 7, "paths": {"evaluation_root": str(tmp_path / "evaluation"), "chunks": str(chunks)}, "datasets": {"bhagavad_gita_qa": {"path": str(source), "version": "test", "languages": ["en"]}}, "split": {"seed": 7}}
    prepare_quick(config)
    row = quick_adapter(config, "quick_bhagavad_gita_qa").load()[0]
    assert row.query == "What does the verse teach about the battlefield?"
