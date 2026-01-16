"""
Fuzzy FCA and RST Reduct Verification
Implements algorithms based on the theoretical definitions in the paper.
"""

import numpy as np
import pandas as pd
from typing import Set, Tuple, List, Dict, Callable
from itertools import product


class ResiduatedLattice:
    """Complete residuated lattice (L, *, ->, 0, 1)"""

    def __init__(self, lattice_type: str = 'godel'):
        """
        Initialize a residuated lattice.

        lattice_type: 'godel', 'goguen', 'lukasiewicz'
        """
        self.lattice_type = lattice_type

        if lattice_type == 'godel':
            # Gödel logic
            self._meet = lambda a, b: min(a, b)
            self._join = lambda a, b: max(a, b)
            self._mult = lambda a, b: min(a, b)
            self._imp = lambda a, b: 1.0 if a <= b else b
        elif lattice_type == 'goguen':
            # Goguen (product) logic
            self._meet = lambda a, b: min(a, b)
            self._join = lambda a, b: max(a, b)
            self._mult = lambda a, b: a * b
            self._imp = lambda a, b: 1.0 if a <= b else b / a if a > 0 else 1.0
        elif lattice_type == 'lukasiewicz':
            # Łukasiewicz logic
            self._meet = lambda a, b: min(a, b)
            self._join = lambda a, b: max(a, b)
            self._mult = lambda a, b: max(0.0, a + b - 1.0)
            self._imp = lambda a, b: min(1.0, 1.0 - a + b)
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type}")

    def meet(self, a: float, b: float) -> float:
        return self._meet(a, b)

    def join(self, a: float, b: float) -> float:
        return self._join(a, b)

    def mult(self, a: float, b: float) -> float:
        """Multiplication operation *"""
        return self._mult(a, b)

    def impl(self, a: float, b: float) -> float:
        """Residuation implication a -> b"""
        return self._imp(a, b)

    def neg(self, a: float) -> float:
        """Negation: ¬a = a -> 0"""
        return self.impl(a, 0.0)

    def double_neg(self, a: float) -> float:
        """Double negation ¬¬a"""
        return self.neg(self.neg(a))


class FuzzyFormalContext:
    """Fuzzy Formal Context (X, Y, φ) for FCA"""

    def __init__(self, data: pd.DataFrame, lattice: ResiduatedLattice = None):
        """
        Initialize fuzzy formal context.

        data: DataFrame with objects as index and attributes as columns
               Values are fuzzy memberships in [0, 1]
        lattice: Residuated lattice (defaults to Godel)
        """
        if lattice is None:
            lattice = ResiduatedLattice('godel')

        self.lattice = lattice
        self.objects = list(data.index)
        self.attributes = list(data.columns)
        self.relation = data.values.astype(float)
        self.object_indices = {obj: i for i, obj in enumerate(self.objects)}
        self.attribute_indices = {attr: i for i, attr in enumerate(self.attributes)}

    def intent(self, mu: Dict[str, float]) -> Dict[str, float]:
        """
        Compute φ^↑(μ): L^X -> (L^Y)^op

        φ^↑(μ)(y) = ⋀_{x∈X} (μ(x) -> φ(x,y))
        """
        result = {}
        for y in self.attributes:
            y_idx = self.attribute_indices[y]
            val = 1.0
            for x in self.objects:
                x_idx = self.object_indices[x]
                if x in mu:
                    val = self.lattice.meet(
                        val,
                        self.lattice.impl(mu[x], self.relation[x_idx, y_idx])
                    )
            result[y] = val
        return result

    def extent(self, lam: Dict[str, float]) -> Dict[str, float]:
        """
        Compute φ^↓(λ): (L^Y)^op -> L^X

        φ^↓(λ)(x) = ⋀_{y∈Y} (λ(y) -> φ(x,y))
        """
        result = {}
        for x in self.objects:
            x_idx = self.object_indices[x]
            val = 1.0
            for y in self.attributes:
                y_idx = self.attribute_indices[y]
                if y in lam:
                    val = self.lattice.meet(
                        val,
                        self.lattice.impl(lam[y], self.relation[x_idx, y_idx])
                    )
            result[x] = val
        return result

    def closure(self, mu: Dict[str, float]) -> Dict[str, float]:
        """
        Compute closure operator: φ^↓φ^↑

        φ^↓φ^↑(μ)(x) = ⋀_{y∈Y} (⋀_{z∈X} (μ(z) -> φ(z,y)) -> φ(x,y))
        """
        intent_mu = self.intent(mu)
        return self.extent(intent_mu)


class FuzzyRoughContext:
    """Fuzzy Context for Rough Set Theory"""

    def __init__(self, data: pd.DataFrame, lattice: ResiduatedLattice = None):
        """
        Initialize fuzzy rough context.

        data: DataFrame with objects as index and attributes as columns
        lattice: Residuated lattice
        """
        if lattice is None:
            lattice = ResiduatedLattice('godel')

        self.lattice = lattice
        self.objects = list(data.index)
        self.attributes = list(data.columns)
        self.relation = data.values.astype(float)
        self.object_indices = {obj: i for i, obj in enumerate(self.objects)}
        self.attribute_indices = {attr: i for i, attr in enumerate(self.attributes)}

    def exists(self, mu: Dict[str, float]) -> Dict[str, float]:
        """
        Compute φ^∃(μ): L^X -> L^Y

        φ^∃(μ)(y) = ⋁_{x∈X} (μ(x) * φ(x,y))
        """
        result = {}
        for y in self.attributes:
            y_idx = self.attribute_indices[y]
            val = 0.0
            for x in self.objects:
                x_idx = self.object_indices[x]
                if x in mu:
                    val = self.lattice.join(
                        val,
                        self.lattice.mult(mu[x], self.relation[x_idx, y_idx])
                    )
            result[y] = val
        return result

    def forall(self, lam: Dict[str, float]) -> Dict[str, float]:
        """
        Compute φ^∀(λ): L^Y -> L^X

        φ^∀(λ)(x) = ⋀_{y∈Y} (φ(x,y) -> λ(y))
        """
        result = {}
        for x in self.objects:
            x_idx = self.object_indices[x]
            val = 1.0
            for y in self.attributes:
                y_idx = self.attribute_indices[y]
                if y in lam:
                    val = self.lattice.meet(
                        val,
                        self.lattice.impl(self.relation[x_idx, y_idx], lam[y])
                    )
            result[x] = val
        return result

    def closure(self, mu: Dict[str, float]) -> Dict[str, float]:
        """
        Compute closure operator: φ^∀φ^∃

        φ^∀φ^∃(μ)(x) = ⋀_{y∈Y} (φ(x,y) -> ⋁_{z∈X} (μ(z) * φ(z,y)))
        """
        exists_mu = self.exists(mu)
        return self.forall(exists_mu)


def generate_fuzzy_subsets(elements: List[str], values: List[float] = None) -> List[Dict[str, float]]:
    """
    Generate all fuzzy subsets from a list of possible membership values.

    elements: list of element names (objects or attributes)
    values: list of possible membership values (e.g., [0, 0.25, 0.5, 0.75, 1])
            If None, defaults to [0, 0.5, 1]
    """
    if values is None:
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
    subsets = []
    for combination in product(values, repeat=len(elements)):
        subsets.append(dict(zip(elements, combination)))
    return subsets


def get_values_from_data(data: pd.DataFrame) -> List[float]:
    """
    Extract unique fuzzy values from data and return sorted list.
    Useful for generating fuzzy subsets that match data granularity.

    data: DataFrame with fuzzy membership values in [0, 1]
    """
    unique_values = sorted(set(data.values.flatten()))
    # Ensure 0 and 1 are included
    if 0.0 not in unique_values:
        unique_values = [0.0] + unique_values
    if 1.0 not in unique_values:
        unique_values = unique_values + [1.0]
    return unique_values


def dict_equal(d1: Dict[str, float], d2: Dict[str, float], tolerance: float = 1e-10) -> bool:
    """Check if two dictionaries are approximately equal"""
    if set(d1.keys()) != set(d2.keys()):
        return False
    return all(abs(d1[k] - d2[k]) < tolerance for k in d1.keys())


def is_fca_reduct(
    full_context: FuzzyFormalContext,
    X_prime: Set[str],
    Y_prime: Set[str],
    verbose: bool = False,
    values: List[float] = None
) -> Tuple[bool, str]:
    """
    Check if (X', Y', φ_{X',Y'}) is a reduct of (X, Y, φ) in FCA.

    According to Definition 4.1.2 and Theorem 4.2.3:
    - Must check: φ^↓φ^↑(μ) = (φ_{X,Y'})^↓(φ_{X,Y'})^↑(μ) for all μ ∈ L^X
    - Must check: φ^↑φ^↓(λ) = (φ_{X',Y})^↑(φ_{X',Y})^↓(λ) for all λ ∈ (L^Y)^op

    Parameters:
    - full_context: FuzzyFormalContext
    - X_prime: subset of objects
    - Y_prime: subset of attributes
    - verbose: print detailed information
    - values: list of fuzzy values to test (e.g., [0, 0.25, 0.5, 0.75, 1])
              If None, uses default [0, 0.5, 1]

    Returns: (is_reduct, explanation)
    """
    # Create subcontext
    sub_data = full_context.relation[
        [full_context.object_indices[x] for x in X_prime], :
    ][:, [full_context.attribute_indices[y] for y in Y_prime]]

    sub_df = pd.DataFrame(
        sub_data,
        index=list(X_prime),
        columns=list(Y_prime)
    )
    sub_context = FuzzyFormalContext(sub_df, full_context.lattice)

    # Check Y reducibility: φ^↓φ^↑(μ) = (φ_{X,Y'})^↓(φ_{X,Y'})^↑(μ)
    # for all μ ∈ L^X
    if verbose:
        print(f"\nChecking Y reducibility for Y' = {Y_prime}")
        print(f"Original Y = {set(full_context.attributes)}")
        print(f"Y \\ Y' = {set(full_context.attributes) - Y_prime}")

    # Generate fuzzy subsets (use provided values or default)
    if values is None:
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_mu = generate_fuzzy_subsets(list(full_context.objects), values)

    if verbose:
        print(f"Testing {len(all_mu)} fuzzy subsets...")

    for i, mu in enumerate(all_mu):
        # LHS: φ^↓φ^↑(μ)
        lhs_closure = full_context.closure(mu)

        # RHS: (φ_{X,Y'})^↓(φ_{X,Y'})^↑(μ)
        sub_context_full_X = FuzzyFormalContext(
            pd.DataFrame(
                full_context.relation[:, [full_context.attribute_indices[y] for y in Y_prime]],
                index=full_context.objects,
                columns=list(Y_prime)
            ),
            full_context.lattice
        )
        rhs_closure = sub_context_full_X.closure(mu)

        if not dict_equal(lhs_closure, rhs_closure):
            if verbose:
                print(f"  Failed at μ #{i}")
                print(f"    LHS (full closure): {lhs_closure}")
                print(f"    RHS (sub closure): {rhs_closure}")
            return False, f"Y reducibility failed at μ #{i}"

    if verbose:
        print("  Y reducibility: PASSED")

    # Check X reducibility: φ^↑φ^↓(λ) = (φ_{X',Y})^↑(φ_{X',Y})^↓(λ)
    # for all λ ∈ (L^Y)^op
    if verbose:
        print(f"\nChecking X reducibility for X' = {X_prime}")
        print(f"Original X = {set(full_context.objects)}")
        print(f"X \\ X' = {set(full_context.objects) - X_prime}")

    all_lam = generate_fuzzy_subsets(list(full_context.attributes), values)

    if verbose:
        print(f"Testing {len(all_lam)} fuzzy subsets...")

    for i, lam in enumerate(all_lam):
        # LHS: φ^↑φ^↓(λ)
        extent_lam = full_context.extent(lam)
        lhs_intent = full_context.intent(extent_lam)

        # RHS: (φ_{X',Y})^↑(φ_{X',Y})^↓(λ)
        sub_context_full_Y = FuzzyFormalContext(
            pd.DataFrame(
                full_context.relation[[full_context.object_indices[x] for x in X_prime], :],
                index=list(X_prime),
                columns=full_context.attributes
            ),
            full_context.lattice
        )
        extent_lam_sub = sub_context_full_Y.extent(lam)
        rhs_intent = sub_context_full_Y.intent(extent_lam_sub)

        if not dict_equal(lhs_intent, rhs_intent):
            if verbose:
                print(f"  Failed at λ #{i}")
                print(f"    LHS (full intent): {lhs_intent}")
                print(f"    RHS (sub intent): {rhs_intent}")
            return False, f"X reducibility failed at λ #{i}"

    if verbose:
        print("  X reducibility: PASSED")

    return True, "Valid FCA reduct"


def is_rst_reduct(
    full_context: FuzzyRoughContext,
    X_prime: Set[str],
    Y_prime: Set[str],
    verbose: bool = False,
    values: List[float] = None
) -> Tuple[bool, str]:
    """
    Check if (X', Y', φ_{X',Y'}) is a reduct of (X, Y, φ) in RST.

    According to Definition 3.2 and Theorem 3.6:
    - Must check: φ^∀φ^∃(μ) = (φ_{X,Y'})^∀(φ_{X,Y'})^∃(μ) for all μ ∈ L^X
    - Must check: φ^∃φ^∀(λ) = (φ_{X',Y})^∃(φ_{X',Y})^∀(λ) for all λ ∈ L^Y

    Parameters:
    - full_context: FuzzyRoughContext
    - X_prime: subset of objects
    - Y_prime: subset of attributes
    - verbose: print detailed information
    - values: list of fuzzy values to test (e.g., [0, 0.25, 0.5, 0.75, 1])
              If None, uses default [0, 0.5, 1]

    Returns: (is_reduct, explanation)
    """
    # Create subcontext
    sub_df = pd.DataFrame(
        full_context.relation[
            [full_context.object_indices[x] for x in X_prime], :
        ][:, [full_context.attribute_indices[y] for y in Y_prime]],
        index=list(X_prime),
        columns=list(Y_prime)
    )
    sub_context = FuzzyRoughContext(sub_df, full_context.lattice)

    # Check Y reducibility: φ^∀φ^∃(μ) = (φ_{X,Y'})^∀(φ_{X,Y'})^∃(μ)
    # for all μ ∈ L^X
    if verbose:
        print(f"\nChecking Y reducibility for Y' = {Y_prime}")
        print(f"Original Y = {set(full_context.attributes)}")

    if values is None:
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_mu = generate_fuzzy_subsets(list(full_context.objects), values)

    if verbose:
        print(f"Testing {len(all_mu)} fuzzy subsets...")

    for i, mu in enumerate(all_mu):
        # LHS: φ^∀φ^∃(μ)
        lhs_closure = full_context.closure(mu)

        # RHS: (φ_{X,Y'})^∀(φ_{X,Y'})^∃(μ)
        sub_context_full_X = FuzzyRoughContext(
            pd.DataFrame(
                full_context.relation[:, [full_context.attribute_indices[y] for y in Y_prime]],
                index=full_context.objects,
                columns=list(Y_prime)
            ),
            full_context.lattice
        )
        rhs_closure = sub_context_full_X.closure(mu)

        if not dict_equal(lhs_closure, rhs_closure):
            if verbose:
                print(f"  Failed at μ #{i}")
                print(f"    LHS: {lhs_closure}")
                print(f"    RHS: {rhs_closure}")
            return False, f"Y reducibility failed at μ #{i}"

    if verbose:
        print("  Y reducibility: PASSED")

    # Check X reducibility: φ^∃φ^∀(λ) = (φ_{X',Y})^∃(φ_{X',Y})^∀(λ)
    # for all λ ∈ L^Y
    if verbose:
        print(f"\nChecking X reducibility for X' = {X_prime}")
        print(f"Original X = {set(full_context.objects)}")

    all_lam = generate_fuzzy_subsets(list(full_context.attributes), values)

    if verbose:
        print(f"Testing {len(all_lam)} fuzzy subsets...")

    for i, lam in enumerate(all_lam):
        # LHS: φ^∃φ^∀(λ)
        forall_lam = full_context.forall(lam)
        lhs_exists = full_context.exists(forall_lam)

        # RHS: (φ_{X',Y})^∃(φ_{X',Y})^∀(λ)
        sub_context_full_Y = FuzzyRoughContext(
            pd.DataFrame(
                full_context.relation[[full_context.object_indices[x] for x in X_prime], :],
                index=list(X_prime),
                columns=full_context.attributes
            ),
            full_context.lattice
        )
        forall_lam_sub = sub_context_full_Y.forall(lam)
        rhs_exists = sub_context_full_Y.exists(forall_lam_sub)

        if not dict_equal(lhs_exists, rhs_exists):
            if verbose:
                print(f"  Failed at λ #{i}")
                print(f"  LHS: {lhs_exists}")
                print(f"  RHS: {rhs_exists}")
            return False, f"X reducibility failed at λ #{i}"

    if verbose:
        print("  X reducibility: PASSED")

    return True, "Valid RST reduct"


def compare_fca_rst_reducts(
    context_data: pd.DataFrame,
    X_prime: Set[str],
    Y_prime: Set[str],
    lattice_type: str = 'godel',
    verbose: bool = False
) -> Dict[str, Tuple[bool, str]]:
    """
    Compare FCA and RST reducts for the same fuzzy context.

    According to Theorem 4.7 (Main Theorem):
    FCA and RST reducts are interdefinable via negation
    if and only if the lattice satisfies double negation law.
    """
    fca_context = FuzzyFormalContext(context_data, ResiduatedLattice(lattice_type))
    rst_context = FuzzyRoughContext(context_data, ResiduatedLattice(lattice_type))

    fca_result = is_fca_reduct(fca_context, X_prime, Y_prime, verbose)
    rst_result = is_rst_reduct(rst_context, X_prime, Y_prime, verbose)

    # Check if lattice satisfies double negation
    lattice = ResiduatedLattice(lattice_type)
    dn_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    has_dn_law = all(abs(lattice.double_neg(a) - a) < 1e-10 for a in dn_values)

    explanation = {
        'has_double_negation': has_dn_law,
        'fca_reduct': fca_result,
        'rst_reduct': rst_result
    }

    # According to Theorem 4.7:
    # If double negation holds, then:
    #   (X', Y', ¬φ_{X',Y'}) is RST reduct of (X, Y, ¬φ)
    #   iff (X', Y', φ_{X',Y'}) is FCA reduct of (X, Y, φ)
    if has_dn_law:
        explanation['agreement'] = (fca_result[0] == rst_result[0])
    else:
        explanation['agreement'] = None
        explanation['note'] = "Double negation law fails, so FCA and RST reducts may differ"

    return explanation
