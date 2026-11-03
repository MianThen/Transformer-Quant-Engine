import re

RULE_ORDER = ["GATE_Full", "B0", "B1", "B2", "B3", "B4"]


def parse_rec(line):
    if not line.startswith("REC "):
        return None
    parts = line.split(" ")
    if len(parts) != 9:
        return None
    _, cell_id_s, rep_id_s, rule_id, selected_s, index_s, abstain_s, fp, reason = parts
    try:
        cell_id = int(cell_id_s)
        rep_id = int(rep_id_s)
        selected = selected_s == "1"
        index = int(index_s)
        abstain = abstain_s == "1"
    except ValueError:
        return None
    return {
        "cell_id": cell_id,
        "rep_id": rep_id,
        "rule_id": rule_id,
        "selected": selected,
        "index": index,
        "abstain": abstain,
        "fp": fp,
        "reason": reason,
    }


def _sort_key(rec):
    rule_id = rec["rule_id"]
    if rule_id in RULE_ORDER:
        rule_idx = RULE_ORDER.index(rule_id)
    else:
        rule_idx = 999
    return (rec["cell_id"], rec["rep_id"], rule_idx, rule_id)


def merge_records(chunks):
    all_records = []
    for chunk in chunks:
        for rec in chunk:
            all_records.append(rec)
    all_records.sort(key=_sort_key)
    return all_records


def aggregate(records):
    agg = {}
    for rec in records:
        rule_id = rec["rule_id"]
        if rule_id not in agg:
            agg[rule_id] = {"n": 0, "selected": 0, "abstain": 0, "rate": 0.0}
        agg[rule_id]["n"] += 1
        if rec["selected"]:
            agg[rule_id]["selected"] += 1
        if rec["abstain"]:
            agg[rule_id]["abstain"] += 1
    for rule_id in agg:
        n = agg[rule_id]["n"]
        if n == 0:
            agg[rule_id]["rate"] = 0.0
        else:
            agg[rule_id]["rate"] = agg[rule_id]["selected"] / n
    return agg


def duplicate_keys(records):
    seen = set()
    for rec in records:
        key = (rec["cell_id"], rec["rep_id"], rec["rule_id"])
        seen.add(key)
    return len(records) - len(seen)


def is_sorted(records):
    for i in range(len(records) - 1):
        if _sort_key(records[i]) > _sort_key(records[i + 1]):
            return False
    return True


if __name__ == "__main__":
    lines = [
        "REC 101 0 GATE_Full 1 0 0 aaaaaaaaaaaaaaaa selected",
        "REC 101 0 B0 0 1 0 aaaaaaaaaaaaaaaa baseline",
        "REC 101 1 GATE_Full 1 0 0 bbbbbbbbbbbbbbbb selected",
        "REC 101 1 B0 1 1 0 bbbbbbbbbbbbbbbb baseline",
        "REC 101 2 GATE_Full 0 2 1 cccccccccccccccc below_threshold",
        "REC 101 2 B0 0 1 0 cccccccccccccccc baseline",
        "REC 205 0 GATE_Full 1 0 0 dddddddddddddddd selected",
        "REC 205 0 B0 0 1 0 dddddddddddddddd baseline",
        "REC 205 1 GATE_Full 0 3 0 eeeeeeeeeeeeeeee below_threshold",
        "REC 205 1 B0 1 3 0 eeeeeeeeeeeeeeee baseline",
        "REC 205 2 GATE_Full 0 0 0 ffffffffffffffff below_threshold",
        "REC 205 2 B0 0 2 0 ffffffffffffffff baseline",
    ]
    records = [parse_rec(line) for line in lines]

    chunk_a = [records[0], records[1], records[4], records[5], records[8], records[9]]
    chunk_b = [records[2], records[3], records[6], records[7], records[10], records[11]]

    merged = merge_records([chunk_a, chunk_b])
    first = merged[0]
    last = merged[-1]
    print(
        "MERGE {} first={},{},{} last={},{},{}".format(
            len(merged),
            first["cell_id"],
            first["rep_id"],
            first["rule_id"],
            last["cell_id"],
            last["rep_id"],
            last["rule_id"],
        )
    )

    agg = aggregate(merged)
    for rule_id in sorted(agg.keys()):
        info = agg[rule_id]
        print(
            "AGG {} n={} selected={} abstain={} rate={:.6f}".format(
                rule_id, info["n"], info["selected"], info["abstain"], info["rate"]
            )
        )

    print("DUP {}".format(duplicate_keys(merged)))

    reversed_merged = list(reversed(merged))
    reshuffled = merge_records([reversed_merged])
    equals = 1 if reshuffled == merged else 0
    print("SHUFFLE_EQUALS {}".format(equals))

    print("REDUCE_DONE")
