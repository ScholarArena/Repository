import math
import runtime

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

try:
    import sympy as sp
    try:
        from sympy.parsing.latex import parse_latex as _parse_latex
    except Exception:
        _parse_latex = None
except Exception:
    sp = None
    _parse_latex = None


def _normalize_params(params):
    return params if isinstance(params, dict) else {}


def _to_float_list(values):
    if values is None:
        return []
    if isinstance(values, (int, float)):
        return [float(values)]
    if not isinstance(values, (list, tuple)):
        return []
    out = []
    for item in values:
        try:
            out.append(float(item))
        except Exception:
            continue
    return out


def _mean(values):
    if np is not None:
        return float(np.mean(values))
    return sum(values) / len(values)


def _variance(values, ddof=1):
    if np is not None:
        return float(np.var(values, ddof=ddof))
    mean_val = _mean(values)
    return sum((v - mean_val) ** 2 for v in values) / max(len(values) - ddof, 1)


def _parse_expression(expr_text, mode):
    if sp is None:
        return None, "sympy_missing"
    expr_text = str(expr_text).strip()
    if not expr_text:
        return None, "missing_expression"
    if mode == "latex":
        if _parse_latex is None:
            return None, "latex_parser_missing"
        try:
            return _parse_latex(expr_text), None
        except Exception as exc:
            return None, f"latex_parse_error:{exc}"
    try:
        return sp.sympify(expr_text), None
    except Exception as exc:
        return None, f"sympify_error:{exc}"


def _dim_add(base, add):
    out = dict(base)
    for key, value in add.items():
        out[key] = out.get(key, 0.0) + value
        if abs(out[key]) < 1e-12:
            out.pop(key, None)
    return out


def _dim_pow(base, exponent):
    return {key: value * exponent for key, value in base.items()}


def _format_dim(dim):
    if not dim:
        return "dimensionless"
    parts = []
    for key in sorted(dim.keys()):
        parts.append(f"{key}^{round(dim[key], 6)}")
    return " ".join(parts)


def _dim_of_expr(expr, dim_map):
    if sp is None:
        return None, "sympy_missing"
    if isinstance(expr, sp.Number):
        return {}, None
    if isinstance(expr, sp.Symbol):
        name = str(expr)
        if name not in dim_map:
            return None, f"unknown_symbol:{name}"
        return {k: float(v) for k, v in (dim_map.get(name) or {}).items()}, None
    if isinstance(expr, sp.Add):
        dims = None
        for arg in expr.args:
            dim, err = _dim_of_expr(arg, dim_map)
            if err:
                return None, err
            if dims is None:
                dims = dim
            elif dims != dim:
                return None, "add_dimension_mismatch"
        return dims or {}, None
    if isinstance(expr, sp.Mul):
        dims = {}
        for arg in expr.args:
            dim, err = _dim_of_expr(arg, dim_map)
            if err:
                return None, err
            dims = _dim_add(dims, dim)
        return dims, None
    if isinstance(expr, sp.Pow):
        base_dim, err = _dim_of_expr(expr.base, dim_map)
        if err:
            return None, err
        exp = expr.exp
        if not exp.is_number:
            return None, "non_numeric_exponent"
        try:
            exp_value = float(exp)
        except Exception:
            return None, "invalid_exponent"
        return _dim_pow(base_dim, exp_value), None
    return {}, None


def _claim_type(claim):
    if not claim:
        return ""
    lower = claim.lower()
    if "significant" in lower and "not" in lower:
        return "not_significant"
    if "significant" in lower:
        return "significant"
    return ""


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        params = _normalize_params(params)
        claim = str(params.get("claim") or "").strip()
        claim_type = _claim_type(claim) or str(params.get("claim_type") or "")
        if not claim_type:
            return runtime.missing(reason="missing_claim", obs_type="significance_check")
        try:
            p_value = float(params.get("p_value"))
        except Exception:
            return runtime.missing(reason="missing_p_value", obs_type="significance_check")
        alpha = float(params.get("alpha") or 0.05)
        if claim_type == "significant":
            supported = p_value < alpha
        else:
            supported = p_value >= alpha
        payload = {
            "claim_type": claim_type,
            "p_value": p_value,
            "alpha": alpha,
            "supported": bool(supported),
        }
        return runtime.ok(payload=payload, prov=[], obs_type="significance_check")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="significance_check")
