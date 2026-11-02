BLOCKS = [
    {"name": "NUL", "cells": 27, "datasets": 27, "reps": 8000, "calls_per_rep": 1},
    {"name": "ALT", "cells": 81, "datasets": 81, "reps": 2000, "calls_per_rep": 1},
    {"name": "MIX", "cells": 12, "datasets": 12, "reps": 2000, "calls_per_rep": 1},
    {"name": "DEC", "cells": 12, "datasets": 12, "reps": 2000, "calls_per_rep": 1},
    {"name": "NEG", "cells":  9, "datasets":  9, "reps": 1000, "calls_per_rep": 1},
    {"name": "GAR", "cells": 18, "datasets": 18, "reps": 1000, "calls_per_rep": 1},
    {"name": "GAU", "cells": 18, "datasets": 18, "reps": 1000, "calls_per_rep": 1},
    {"name": "FRQ", "cells": 12, "datasets": 12, "reps": 1000, "calls_per_rep": 1},
    {"name": "SVL", "cells":  4, "datasets":  4, "reps": 1000, "calls_per_rep": 1},
    {"name": "BYA", "cells":  4, "datasets":  0, "reps": 2000, "calls_per_rep": 1},
    {"name": "RET_K1",  "cells": 1, "datasets": 1, "reps": 2000, "calls_per_rep": 1},
    {"name": "RET_K10", "cells": 1, "datasets": 0, "reps": 2000, "calls_per_rep": 10},
    {"name": "ABL", "cells": 40, "datasets":  0, "reps": 1000, "calls_per_rep": 1},
    {"name": "SWP", "cells": 30, "datasets":  0, "reps": 2000, "calls_per_rep": 1},
]


def derive_counts(blocks):
    cells = sum(b["cells"] for b in blocks)
    datasets = sum(b["datasets"] for b in blocks)
    calls = sum(b["cells"] * b["reps"] * b["calls_per_rep"] for b in blocks)
    return (cells, datasets, calls)


def next_rep_step(reps):
    table = {8000: 4000, 2000: 1000, 1000: 500}
    return table.get(reps, reps)


def plan_reps(units, cost, threshold):
    result = []
    for u in units:
        new_u = dict(u)
        if u["N"] == 3:
            new_u["reps"] = u["reps"]
        else:
            c = cost[(u["T"], u["N"])]
            if c > threshold:
                new_u["reps"] = next_rep_step(u["reps"])
            else:
                new_u["reps"] = u["reps"]
        result.append(new_u)
    return result


def total_calls(units):
    return sum(u["reps"] * u.get("calls_per_rep", 1) for u in units)


if __name__ == "__main__":
    c, d, n = derive_counts(BLOCKS)
    print(f"DERIVE real cells={c} datasets={d} calls={n}")

    mutated = [b for b in BLOCKS if b["name"] != "NEG"]
    c2, d2, n2 = derive_counts(mutated)
    print(f"DERIVE mutated cells={c2} datasets={d2} calls={n2}")

    ladder_parts = []
    for r in (8000, 2000, 1000, 4000, 500):
        ladder_parts.append(f"{r}->{next_rep_step(r)}")
    print("LADDER " + " ".join(ladder_parts))

    units = [
        {"T": 4,   "N": 126, "reps": 8000, "calls_per_rep": 1},
        {"T": 3,   "N": 3,   "reps": 2000, "calls_per_rep": 1},
        {"T": 100, "N": 266, "reps": 2000, "calls_per_rep": 1},
        {"T": 20,  "N": 266, "reps": 1000, "calls_per_rep": 1},
        {"T": 100, "N": 520, "reps": 1000, "calls_per_rep": 1},
    ]
    cost = {(4, 126): 10.0, (3, 3): 99.0, (100, 266): 0.5, (20, 266): 50.0, (100, 520): 0.5}

    planned = plan_reps(units, cost, 1.0)
    for i, u in enumerate(planned, 1):
        print(f"PLAN u{i} {u['reps']}")
    print(f"TOTAL {total_calls(planned)}")

    planned2 = plan_reps(units, cost, 100.0)
    for i, u in enumerate(planned2, 1):
        print(f"PLAN2 u{i} {u['reps']}")
    print(f"TOTAL2 {total_calls(planned2)}")

    print("BUDGET_DONE")
