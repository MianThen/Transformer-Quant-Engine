import copy

METRICS = {
    "fpr": {"GATE": {"T4": 0.9435, "T20": 0.1650, "T100": 0.0350}},
    "abstain": {"T4": 0.21, "T20": 0.002, "T100": 0.0},
}


def get_path(d, path):
    """按 'a.b.c' 逐段取 d['a']['b']['c'] 并返回。取不到就抛 KeyError（不要吞掉）。"""
    parts = path.split(".")
    cur = d
    for p in parts:
        cur = cur[p]
    return cur


def build_plot_specs(metrics):
    """返回两个 spec（顺序固定）：
       spec0: name="fpr_vs_t"，一条 series，label="GATE"，
              3 个点，对应 T4/T20/T100，source 分别为 "fpr.GATE.T4" 等，
              value 由 get_path 取得。
       spec1: name="abstain_vs_t"，一条 series，label="abstain"，
              3 个点，source 分别为 "abstain.T4" 等。
       每个点的形状：{"value": <float>, "source": "<路径字符串>"}
       spec 的形状：{"name": str, "series": [{"label": str, "points": [点...]}]}"""
    specs = []

    # spec0: fpr_vs_t
    points0 = []
    for t in ["T4", "T20", "T100"]:
        src = f"fpr.GATE.{t}"
        val = get_path(metrics, src)
        points0.append({"value": val, "source": src})
    specs.append({
        "name": "fpr_vs_t",
        "series": [{"label": "GATE", "points": points0}]
    })

    # spec1: abstain_vs_t
    points1 = []
    for t in ["T4", "T20", "T100"]:
        src = f"abstain.{t}"
        val = get_path(metrics, src)
        points1.append({"value": val, "source": src})
    specs.append({
        "name": "abstain_vs_t",
        "series": [{"label": "abstain", "points": points1}]
    })

    return specs


def trace(specs, metrics):
    """逐 spec、逐 series、逐 point 检查：
         abs(get_path(metrics, point["source"]) - point["value"]) <= 1e-12
       全部通过返回 (True, "")；
       第一个不通过的点返回 (False, <该 spec 的 name>)。
       【必须是严格相等语义：一旦有人把 value 写死成别的数，这里就要失败。】"""
    for spec in specs:
        for series in spec["series"]:
            for point in series["points"]:
                actual = get_path(metrics, point["source"])
                if abs(actual - point["value"]) > 1e-12:
                    return (False, spec["name"])
    return (True, "")


if __name__ == "__main__":
    specs = build_plot_specs(METRICS)

    # 第 1 步
    for i, spec in enumerate(specs):
        total_points = sum(len(s["points"]) for s in spec["series"])
        print(f"PLOT {i} {spec['name']} {total_points}")

    # 第 2 步
    ok, _ = trace(specs, METRICS)
    print(f"TRACE_OK {1 if ok else 0}")

    # 第 3 步
    modified = copy.deepcopy(METRICS)
    modified["fpr"]["GATE"]["T20"] = 0.1651
    ok_bad, name_bad = trace(specs, modified)
    print(f"TRACE_BAD {1 if ok_bad else 0} {name_bad}")

    # 第 4 步
    print("PLOT_DONE")
