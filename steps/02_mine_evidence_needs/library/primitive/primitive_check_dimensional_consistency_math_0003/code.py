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


def _dim_of_ast(node, dim_map):
    if isinstance(node, ast.Constant):
        return {}, None
    if isinstance(node, ast.Num):
        return {}, None
    if isinstance(node, ast.Name):
        name = node.id
        if name not in dim_map:
            return None, f"unknown_symbol:{name}"
        return {k: float(v) for k, v in (dim_map.get(name) or {}).items()}, None
    if isinstance(node, ast.UnaryOp):
        return _dim_of_ast(node.operand, dim_map)
    if isinstance(node, ast.BinOp):
        left_dim, err = _dim_of_ast(node.left, dim_map)
        if err:
            return None, err
        right_dim, err = _dim_of_ast(node.right, dim_map)
        if err:
            return None, err
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if left_dim != right_dim:
                return None, "add_dimension_mismatch"
            return left_dim, None
        if isinstance(node.op, ast.Mult):
            return _dim_add(left_dim, right_dim), None
        if isinstance(node.op, ast.Div):
            return _dim_add(left_dim, _dim_pow(right_dim, -1.0)), None
        if isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant):
                exp_value = node.right.value
            elif isinstance(node.right, ast.Num):
                exp_value = node.right.n
            else:
                return None, "non_numeric_exponent"
            try:
                exp_value = float(exp_value)
            except Exception:
                return None, "invalid_exponent"
            return _dim_pow(left_dim, exp_value), None
        return None, "unsupported_expression"
    return None, "unsupported_expression"

def _dim_of_expr(expr, dim_map):
    if sp is not None and isinstance(expr, sp.Basic):
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
    if isinstance(expr, ast.AST):
        return _dim_of_ast(expr, dim_map)
    return None, "unsupported_expression"


def execute(context, params, primitives=None, controlled_llm=None):
    try:
        params = _normalize_params(params)
        lhs = params.get("lhs")
        rhs = params.get("rhs")
        if lhs is None or rhs is None:
            return runtime.missing(error="missing_lhs_or_rhs", obs_type="dimension_check")
        mode = str(params.get("format") or "sympy")
        dim_map = params.get("dimensions") or {}
        expr_lhs, err_lhs = _parse_expression(lhs, mode)
        if err_lhs:
            return runtime.fail(error=err_lhs, obs_type="dimension_check")
        expr_rhs, err_rhs = _parse_expression(rhs, mode)
        if err_rhs:
            return runtime.fail(error=err_rhs, obs_type="dimension_check")
        dim_lhs, err_dim_lhs = _dim_of_expr(expr_lhs, dim_map)
        if err_dim_lhs:
            return runtime.fail(error=err_dim_lhs, obs_type="dimension_check")
        dim_rhs, err_dim_rhs = _dim_of_expr(expr_rhs, dim_map)
        if err_dim_rhs:
            return runtime.fail(error=err_dim_rhs, obs_type="dimension_check")
        consistent = dim_lhs == dim_rhs
        payload = {
            "lhs_dim": _format_dim(dim_lhs),
            "rhs_dim": _format_dim(dim_rhs),
            "consistent": bool(consistent),
        }
        return runtime.ok(payload=payload, prov=[], obs_type="dimension_check")
    except Exception as exc:
        return runtime.fail(error=str(exc), obs_type="dimension_check")
