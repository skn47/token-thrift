from tokenthrift.corpus.documents import load_all_chunks
from tokenthrift.corpus.labels import load_question_labels
from tokenthrift.corpus.registry import resolve_corpus


def test_bundled_corpora_have_disjoint_doc_and_chunk_ids():
    lighthouse = resolve_corpus("lighthouse")
    nimbus = resolve_corpus("nimbus")
    lh_chunks = load_all_chunks(lighthouse.corpus_dir)
    nb_chunks = load_all_chunks(nimbus.corpus_dir)

    assert {c.doc_id for c in lh_chunks}.isdisjoint({c.doc_id for c in nb_chunks})
    assert {c.chunk_id for c in lh_chunks}.isdisjoint(
        {c.chunk_id for c in nb_chunks})


def test_bundled_corpora_have_disjoint_question_ids():
    lighthouse = resolve_corpus("lighthouse")
    nimbus = resolve_corpus("nimbus")
    lh_qids = {q.question_id for q in load_question_labels(lighthouse.labels_path)}
    nb_qids = {q.question_id for q in load_question_labels(nimbus.labels_path)}
    assert lh_qids.isdisjoint(nb_qids)


def test_artifacts_live_under_separate_corpus_directories():
    lighthouse = resolve_corpus("lighthouse")
    nimbus = resolve_corpus("nimbus")
    assert lighthouse.artifacts_dir != nimbus.artifacts_dir
    assert lighthouse.artifacts_dir.name == "lighthouse"
    assert nimbus.artifacts_dir.name == "nimbus"
