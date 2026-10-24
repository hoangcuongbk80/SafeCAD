import csv
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pytest
from PIL import Image
from safecad.data import main, read_metadata, safe_extract, physical_sample


def realiad_fixture(root):
    category = "audiojack"
    records = []
    for label, count in ((0,20), (1,6)):
        for sample in range(count):
            for view in ("C1", "C2"):
                status = "NG" if label else "OK"
                stem = f"audiojack_{sample:04d}_{status}_{view}_000000"
                relative = Path(status)/f"S{sample:04d}"/(stem+".jpg")
                path = root/"realiad_1024"/category/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((8,12,3), sample*5, np.uint8)).save(path)
                row = dict(image_path=relative.as_posix(), anomaly_class="defect" if label else "OK", mask_path="")
                if label:
                    mask = relative.with_name(stem+"_mask.png")
                    pixels = np.zeros((8,12), np.uint8)
                    pixels[2:4,3:5] = 255
                    Image.fromarray(pixels).save(root/"realiad_1024"/category/mask)
                    row["mask_path"] = mask.as_posix()
                records.append(row)
    metadata = root/"realiad_jsons"
    metadata.mkdir()
    (metadata/f"{category}.json").write_text(json.dumps(dict(train=records[:10], test=records[10:])))


def prepare(root, output, contamination):
    main(["prepare-realiad", "--root", str(root), "--output", str(output), "--partition", "all",
          "--initialization", "2", "--calibration", "2", "--stream-size", "8", "--test-normal", "4",
          "--test-anomalous", "3", "--contamination", str(contamination)])
    return {name: json.loads((output/f"{name}.json").read_text())
            for name in ("commission", "adaptation", "adaptation_labels", "evaluation", "protocol")}


def test_realiad_physical_splits_no_label_leakage_and_fixed_test_cohort(tmp_path):
    root = tmp_path/"Real-IAD"
    realiad_fixture(root)
    clean_dir, noisy_dir = tmp_path/"clean", tmp_path/"noisy"
    clean = prepare(root, clean_dir, 0)
    noisy = prepare(root, noisy_dir, 0.25)
    assert len(noisy["adaptation"]) == 8
    assert sum(r["label"] for r in noisy["adaptation_labels"]) == 2
    assert all("label" not in r and "mask" not in r for r in noisy["adaptation"])
    assert len(noisy["evaluation"]) == 7
    clean_test = {(clean_dir/r["image"]).resolve() for r in clean["evaluation"]}
    noisy_test = {(noisy_dir/r["image"]).resolve() for r in noisy["evaluation"]}
    assert clean_test == noisy_test
    domain = noisy["commission"]["domains"][0]
    init = {physical_sample(p) for p in domain["initialization"]}
    cal = {physical_sample(p) for p in domain["calibration"]}
    adapt = {r["sample_id"] for r in noisy["adaptation"]}
    test = {r["sample_id"] for r in noisy["evaluation"]}
    assert len(init | cal | adapt | test) == len(init)+len(cal)+len(adapt)+len(test)
    assert noisy["protocol"]["realized_contamination"] == 0.25
    assert min(r["episode"] for r in noisy["evaluation"]) > max(r["episode"] for r in noisy["adaptation"])


def test_json_lines_metadata_supported(tmp_path):
    path = tmp_path/"category.json"
    path.write_text('{"filename":"a.jpg","label":0}\n{"filename":"b.jpg","label":1}\n')
    assert len(read_metadata(path)) == 2


def test_archive_extraction_handles_category_wrapper(tmp_path):
    archive = tmp_path/"audiojack.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("realiad_1024/audiojack/OK/test.jpg", b"contents")
    safe_extract(archive, tmp_path/"images", "audiojack", "1024")
    assert (tmp_path/"images/audiojack/OK/test.jpg").read_bytes() == b"contents"


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt", "C:/escape.txt", "folder/../../escape.txt"])
def test_archive_extraction_rejects_traversal(tmp_path, name):
    archive = tmp_path/"bad.zip"
    with ZipFile(archive, "w") as z:
        z.writestr(name, b"bad")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path/"images")


def test_coal_index_preparation_and_session_split_guard(tmp_path):
    image = tmp_path/"image.jpg"
    Image.fromarray(np.zeros((8,8,3), np.uint8)).save(image)
    # Distinct paths identify distinct acquisition frames.
    rows = []
    for i, (split, timestamp) in enumerate((("initialization",0), ("calibration",60), ("calibration",120),
                                         ("adaptation",180), ("adaptation",180.2), ("evaluation",300))):
        target = tmp_path/f"{i}.jpg"
        target.write_bytes(image.read_bytes())
        rows.append(dict(image=target.name, domain="C1", session="commission" if i < 3 else split,
                         timestamp=timestamp, split=split, label="0" if i < 3 else "", mask=""))
    index = tmp_path/"index.csv"
    def write_rows():
        with index.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    write_rows()
    output = tmp_path/"prepared"
    main(["prepare-coal", "--index", str(index), "--output", str(output), "--calibration-min", "2"])
    adapt = json.loads((output/"adaptation.json").read_text())
    assert adapt[0]["episode"] == adapt[1]["episode"]
    assert "label" not in adapt[0]
    rows[-1]["session"] = "adaptation"
    write_rows()
    with pytest.raises(ValueError, match="Session crosses"):
        main(["prepare-coal", "--index", str(index), "--output", str(output), "--calibration-min", "2"])
