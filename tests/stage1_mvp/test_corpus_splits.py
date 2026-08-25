from tokenthrift.corpus.documents import load_documents
from tokenthrift.corpus.labels import load_question_labels
from tokenthrift.corpus.splits import load_persisted_splits


def test_no_chunk_overlap_across_splits():
    splits = load_persisted_splits()
    train_docs = set(splits.doc_ids_for("train"))
    val_docs = set(splits.doc_ids_for("val"))
    test_docs = set(splits.doc_ids_for("test"))

    assert train_docs & val_docs == set()
    assert train_docs & test_docs == set()
    assert val_docs & test_docs == set()

    documents = load_documents()
    assert train_docs | val_docs | test_docs == set(documents.keys())


def test_every_split_has_documents_from_more_than_one_type():
    documents = load_documents()
    splits = load_persisted_splits()
    for split in ("train", "val", "test"):
        doc_ids = splits.doc_ids_for(split)
        assert doc_ids, f"{split} split has no documents"
        types = {documents[d]["doc_type"] for d in doc_ids}
        assert len(types) > 1, f"{split} split only contains doc_type {types}"


def test_cross_split_questions_are_explicitly_excluded_not_silently_assigned():
    splits = load_persisted_splits()
    labels_by_id = {q.question_id: q for q in load_question_labels()}

    assert splits.excluded_question_ids, (
        "expected at least one cross-document question to exercise the "
        "exclusion path")

    for qid in splits.excluded_question_ids:
        q = labels_by_id[qid]
        touched_splits = {
            splits.doc_split[chunk_id.split("::", 1)[0]]
            for chunk_id in q.relevant_chunk_ids
        }
        assert len(touched_splits) > 1, (
            f"{qid} was excluded but all its evidence is in one split")

    assert splits.excluded_question_ids.isdisjoint(set(splits.question_split))
