import ast
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

def _latex_to_python(expr_text):
    text = str(expr_text)
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("^", "**")
    text = text.replace("{", "(").replace("}", ")")
    return text

def _ast_is_safe(node):
    if isinstance(node, ast.Expression):
        return _ast_is_safe(node.body)
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            return False
        return _ast_is_safe(node.left) and _ast_is_safe(node.right)
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.UAdd, ast.USub)) and _ast_is_safe(node.operand)
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))
    if isinstance(node, ast.Num):
        return True
    return False

def _parse_ast_expression(expr_text):
    try:
        tree = ast.parse(expr_text, mode="eval")
    except Exception as exc:
        return None, f"ast_parse_error:{exc}"
    if not _ast_is_safe(tree):
        return None, "unsupported_expression"
    return tree.body, None

def _ast_canonical(node):
    if isinstance(node, ast.BinOp):
        left = _ast_canonical(node.left)
        right = _ast_canonical(node.right)
        op_name = type(node.op).__name__
        if isinstance(node.op, (ast.Add, ast.Mult)):
            items = sorted([left, right], key=repr)
            return (op_name, items[0], items[1])
        return (op_name, left, right)
    if isinstance(node, ast.UnaryOp):
        return (type(node.op).__name__, _ast_canonical(node.operand))
    if isinstance(node, ast.Name):
        return ("Name", node.id)
    if isinstance(node, ast.Constant):
        return ("Const", node.value)
    if isinstance(node, ast.Num):
        return ("Num", node.n)
    return ("Unknown", type(node).__name__)

def _ast_repr(node):
    return repr(_ast_canonical(node))

def _expr_repr(expr):
    if sp is not None and isinstance(expr, sp.Basic):
        return sp.srepr(expr)
    if isinstance(expr, ast.AST):
        return _ast_repr(expr)
    return str(expr)


def _parse_expression(expr_text, mode):
    expr_text = str(expr_text).strip()
    if not expr_text:
        return None, "missing_expression"
    if sp is not None:
        if mode == "latex":
            if _parse_latex is not None:
                try:
                    return _parse_latex(expr_text), None
                except Exception:
                    pass
        else:
            try:
                return sp.sympify(expr_text), None
            except Exception:
                pass
    if mode == "latex":
        expr_text = _latex_to_python(expr_text)
    return _parse_ast_expression(expr_text)


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
        eq_a = params.get("eq_a") or ""
        eq_b = params.get("eq_b") or ""
        mode = str(params.get("format") or "latex")
        semantic = bool(params.get("semantic"))
        expr_a, err_a = _parse_expression(eq_a, mode)
        if err_a:
            return runtime.fail(error=err_a, obs_type="equation_compare")
        expr_b, err_b = _parse_expression(eq_b, mode)
        if err_b:
            return runtime.fail(error=err_b, obs_type="equation_compare")
        rep_a = _expr_repr(expr_a)
        rep_b = _expr_repr(expr_b)
        match = rep_a == rep_b
        semantic_match = None
        if semantic:
            if sp is not None and isinstance(expr_a, sp.Basic) and isinstance(expr_b, sp.Basic):
                try:
                    semantic_match = bool(sp.simplify(expr_a - expr_b) == 0)
                except Exception:
                    semantic_match = False
            else:
                semantic_match = bool(match)
        payload = {
            "match": bool(match),
            "semantic_match": semantic_match,
            "srepr_a": rep_a,
            "srepr_b": rep_b,
        }
        return runtime.ok(payload=payload, prov=[], obs_type="equation_compare")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="equation_compare")
