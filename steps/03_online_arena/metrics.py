from typing import Any, Dict, Iterable, List


def compute_metrics(events: Iterable[Dict[str, Any]], threads: List[Dict[str, Any]]) -> Dict[str, Any]:
    moves = [e for e in events if e.get("type") == "move"]
    total_moves = len(moves)
    ok_moves = sum(1 for m in moves if (m.get("observation") or {}).get("status") == "ok")
    missing_or_fail = sum(1 for m in moves if (m.get("observation") or {}).get("status") != "ok")
    halluc_missing = 0
    total_claims = 0
    supported_claims = 0
    plan_total = 0
    plan_correct = 0
    for m in moves:
        observation = m.get("observation") or {}
        action = m.get("action") or {}
        claims = action.get("claims") or []
        total_claims += len(claims)
        for claim in claims:
            cites = claim.get("cites") or []
            if cites:
                supported_claims += 1
        if observation.get("status") != "ok" and claims:
            halluc_missing += 1
        gold = m.get("gold_plan")
        if gold:
            plan_total += 1
            planned = m.get("plan") or {}
            if planned.get("intent") == gold.get("intent") and (planned.get("skill_call") or {}).get("name") == (gold.get("skill_call") or {}).get("name"):
                plan_correct += 1
    closed_threads = 0
    for thread in threads:
        phase = thread.get("phase")
        requests = (thread.get("ledger") or {}).get("requests") or []
        if phase == "Closed" and not requests:
            closed_threads += 1
    metrics = {
        "total_moves": total_moves,
        "ok_rate": ok_moves / total_moves if total_moves else 0.0,
        "missing_or_fail_rate": missing_or_fail / total_moves if total_moves else 0.0,
        "halluc_missing_rate": halluc_missing / total_moves if total_moves else 0.0,
        "close_rate": closed_threads / len(threads) if threads else 0.0,
        "claim_support_proxy": supported_claims / total_claims if total_claims else 0.0,
    }
    if plan_total:
        metrics["plan_acc"] = plan_correct / plan_total
    else:
        metrics["plan_acc"] = None
    return metrics
