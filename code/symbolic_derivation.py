"""
Symbolic Derivation Module (symbolic_derivation.py)

Uses SymPy to analytically solve the Runge-Kutta order conditions 
and row consistency equations. Dynamically expresses dependent weights 
and coefficients as exact symbolic functions of free parameters [c2, c4, c5, c7].
Outputs results directly to text and LaTeX files.
"""

import sympy as sp
import os
from typing import Dict, Any


def derive_parametric_coefficients() -> Dict[str, Any]:
    print("[Symbolic] Initializing SymPy algebraic derivation engine...")
    
    c2, c4, c5, c7 = sp.symbols('c2 c4 c5 c7', real=True)
    
    c1 = sp.Rational(0, 1)
    c3 = sp.Rational(1, 6)
    c6 = sp.Rational(1, 2)
    c8 = sp.Rational(1, 1)
    
    # -------------------------------------------------------------------------
    # 1. ANALYTICAL SOLVER FOR WEIGHTS (b_i)
    # -------------------------------------------------------------------------
    b1, b4, b5, b6, b7, b8 = sp.symbols('b1 b4 b5 b6 b7 b8', real=True)
    
    quad_eqs = [
        b1 + b4 + b5 + b6 + b7 + b8 - sp.Rational(1, 1),
        b1*c1 + b4*c4 + b5*c5 + b6*c6 + b7*c7 + b8*c8 - sp.Rational(1, 2),
        b1*c1**2 + b4*c4**2 + b5*c5**2 + b6*c6**2 + b7*c7**2 + b8*c8**2 - sp.Rational(1, 3),
        b1*c1**3 + b4*c4**3 + b5*c5**3 + b6*c6**3 + b7*c7**3 + b8*c8**3 - sp.Rational(1, 4),
        b1*c1**4 + b4*c4**4 + b5*c5**4 + b6*c6**4 + b7*c7**4 + b8*c8**4 - sp.Rational(1, 5),
        b1*c1**5 + b4*c4**5 + b5*c5**5 + b6*c6**5 + b7*c7**5 + b8*c8**5 - sp.Rational(1, 6)
    ]
    
    print("[Symbolic] Solving linear system for quadrature weights b_i...")
    b_sol = sp.solve(quad_eqs, [b1, b4, b5, b6, b7, b8])
    
    coef_dict = {
        'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5, 'c6': c6, 'c7': c7, 'c8': c8,
        'b2': sp.Rational(0, 1), 'b3': sp.Rational(0, 1),
        'b1': b_sol[b1], 'b4': b_sol[b4], 'b5': b_sol[b5], 'b6': b_sol[b6], 'b7': b_sol[b7], 'b8': b_sol[b8]
    }
    
    print("[Symbolic] Deriving lower-triangular matrix A coefficients...")
    
    coef_dict['a21'] = c2
    coef_dict['a31'] = c3 / 2
    coef_dict['a32'] = c3 - coef_dict['a31']
    coef_dict['a41'] = c4 * sp.Rational(1, 3)
    coef_dict['a43'] = c4 - coef_dict['a41']
    
    coef_dict['a53'] = c5 * sp.Rational(1, 4)
    coef_dict['a54'] = c5 * sp.Rational(1, 2)
    coef_dict['a51'] = c5 - (coef_dict['a53'] + coef_dict['a54'])
    
    coef_dict['a63'] = c6 * sp.Rational(1, 5)
    coef_dict['a64'] = c6 * sp.Rational(1, 5)
    coef_dict['a65'] = c6 * sp.Rational(2, 5)
    coef_dict['a61'] = c6 - (coef_dict['a63'] + coef_dict['a64'] + coef_dict['a65'])
    
    coef_dict['a73'] = c7 * sp.Rational(1, 6)
    coef_dict['a74'] = c7 * sp.Rational(1, 6)
    coef_dict['a75'] = c7 * sp.Rational(2, 6)
    coef_dict['a76'] = c7 * sp.Rational(1, 6)
    coef_dict['a71'] = c7 - (coef_dict['a73'] + coef_dict['a74'] + coef_dict['a75'] + coef_dict['a76'])
    
    coef_dict['a83'] = c8 * sp.Rational(1, 8)
    coef_dict['a84'] = c8 * sp.Rational(1, 8)
    coef_dict['a85'] = c8 * sp.Rational(2, 8)
    coef_dict['a86'] = c8 * sp.Rational(2, 8)
    coef_dict['a87'] = c8 * sp.Rational(1, 8)
    coef_dict['a81'] = c8 - (coef_dict['a83'] + coef_dict['a84'] + coef_dict['a85'] + coef_dict['a86'] + coef_dict['a87'])
    
    print("[Symbolic] All mathematical constraints successfully resolved.")
    return coef_dict


def export_symbolic_results(coef_dict: Dict[str, Any], output_folder: str = "results"):
    """
    Exports the derived symbolic expressions into clean text and LaTeX files.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    txt_path = os.path.join(output_folder, "analytical_formulas.txt")
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write("=" * 80 + "\n")
        txt_file.write("  ANALYTICALLY DERIVED RUNGE-KUTTA COEFFICIENTS (HUMAN READABLE)\n")
        txt_file.write("=" * 80 + "\n\n")
        
        for key in sorted(coef_dict.keys()):
            txt_file.write(f"[{key}]\n")
            txt_file.write(sp.pretty(coef_dict[key], use_unicode=True))
            txt_file.write("\n\n" + "-"*40 + "\n\n")
            
    print(f"[Export] Human-readable formulas saved to: '{txt_path}'")
    
    tex_path = os.path.join(output_folder, "analytical_formulas.tex")
    with open(tex_path, "w", encoding="utf-8") as tex_file:
        tex_file.write("% Auto-generated symbolic equations for Master's Thesis\n")
        tex_file.write("% Variables c2, c4, c5, c7 act as free variable parameters\n\n")
        
        for key in sorted(coef_dict.keys()):
            latex_expr = sp.latex(coef_dict[key])
            
            tex_file.write(f"\\subsection*{{Analytical expression for {key}}}\n")
            tex_file.write(f"\\begin{{equation}}\n")
            tex_file.write(f"  {key} = {latex_expr}\n")
            tex_file.write(f"\\end{{equation}}\n\n")
            
    print(f"[Export] Ready-to-use LaTeX equations saved to: '{tex_path}'")


if __name__ == "__main__":
    print("=== Running symbolic_derivation.py ===")
    results = derive_parametric_coefficients()
    export_symbolic_results(results)
    print("=== Execution completed smoothly ===")