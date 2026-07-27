from src.importing.grouping import suggest_groups

def image(barcode=False, date=False, product=False):
    return {"evidence_types": {"BARCODE": barcode, "DATE_LABEL": date, "PRODUCT_IDENTITY": product}}

def test_one_image_group_when_all_evidence_exists():
    groups = suggest_groups([image(True, True, True)])
    assert [(group.start, group.end, group.confidence) for group in groups] == [(0, 0, "HIGH")]

def test_three_image_protocol_group():
    groups = suggest_groups([image(True), image(False, True), image(False, False, True)])
    assert [(group.start, group.end) for group in groups] == [(0, 2)]

def test_next_barcode_protects_boundary():
    groups = suggest_groups([image(True), image(True, True, True)])
    assert [(group.start, group.end) for group in groups] == [(0, 0), (1, 1)]
