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


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        params = _normalize_params(params)
        group_a = _to_float_list(params.get("group_a"))
        group_b = _to_float_list(params.get("group_b"))
        if len(group_a) < 2 or len(group_b) < 2:
            return runtime.missing(reason="insufficient_samples", obs_type="effect_size")
        mean_a = _mean(group_a)
        mean_b = _mean(group_b)
        var_a = _variance(group_a, ddof=1)
        var_b = _variance(group_b, ddof=1)
        n_a = len(group_a)
        n_b = len(group_b)
        pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
        if pooled <= 0:
            return runtime.missing(reason="zero_variance", obs_type="effect_size")
        pooled_std = math.sqrt(pooled)
        effect = (mean_a - mean_b) / pooled_std
        precision = int(params.get("round") or 6)
        payload = {
            "method": "cohen_d",
            "effect_size": round(effect, precision),
            "mean_a": round(mean_a, precision),
            "mean_b": round(mean_b, precision),
            "std_a": round(math.sqrt(var_a), precision),
            "std_b": round(math.sqrt(var_b), precision),
            "n_a": n_a,
            "n_b": n_b,
        }
        return runtime.ok(payload=payload, prov=[], obs_type="effect_size")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="effect_size")
