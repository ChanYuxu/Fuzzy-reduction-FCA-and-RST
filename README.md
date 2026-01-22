# Fuzzy-reduction-FCA-and-RST
Repository for algorithms implementing attribute reduction via Fuzzy Formal Concept Analysis (FCA) and Fuzzy Rough Set Theory (RST).

---

# UCI Wine Dataset - Reduct Testing

## Dataset Information
The UCI Wine Dataset consists of 178 samples characterized by 13 chemical constituents (attributes) and 1 categorical label (representing 3 types of cultivars).

### Attribute List:
1. alcohol
2. malic_acid
3. ash
4. alcalinity_of_ash
5. magnesium
6. total_phenols
7. flavanoids
8. nonflavanoid_phenols
9. proanthocyanins
10. color_intensity
11. hue
12. od280/od315_of_diluted_wines
13. proline
14. class (Decision attribute)

---

## Testing Workflow

### 1. Data Preprocessing
**Script:** `wine_reduct_simple.py`

1. **Sampling:** Randomly selects samples from different classes.
2. **Normalization:** Performs Min-Max normalization on all numerical attributes to the range [0, 1].
3. **Discretization:** Rounds normalized values to a precision of 0.25 (i.e., 0.0, 0.25, 0.5, 0.75, 1.0).
4. **Cleaning:** Removes the `class` column, retaining only numerical attributes for reduction testing.

### 2. Reduction Testing
Utilizes core functions from `fuzzy_fca_reduct.py`:

* `FuzzyFormalContext`: Creates a fuzzy formal context (for FCA).
* `FuzzyRoughContext`: Creates a fuzzy rough context (for RST).
* `is_fca_reduct()`: Validates if a specific attribute subset is an FCA reduct.
* `is_rst_reduct()`: Validates if a specific attribute subset is an RST reduct.

### 3. Methodology
**Single Attribute Removal:**
* Sequentially removes one attribute and checks if the remaining set $Y'$ maintains the reductive properties of the original context.
* For each subset $Y'$:
  * $X'$ = All objects
  * $Y'$ = All attributes - {removed_attribute}

**Combination Attribute Removal:**
* Tests the removal of attribute pairs.
* Due to computational complexity, the number of combinations tested is limited.

---

## Theoretical Background & Expected Results

According to **Theorem 4.7** in the source paper:

### Case A: Gödel Residuated Lattice
The Gödel lattice does **not** satisfy the Law of Double Negation ($\neg \neg 0.5 = 1 \neq 0.5$).
* **Expectation:** FCA and RST reduction results may differ.
* **Result:** Potential for **divergence** between the two theories.

### Case B: Łukasiewicz Residuated Lattice
The Łukasiewicz lattice satisfies the Law of Double Negation ($\neg \neg a = a$ for all $a \in [0, 1]$).
* **Expectation:** FCA and RST reduction results should be consistent.
* **Result:** No divergence should occur.

---

## Technical Details

### Fuzzy Value Precision
The normalized values are mapped to a 5-level fuzzy scale:
$$normalized = \frac{x - min}{max - min}$$
$$rounded\_to\_025 = \frac{round(normalized \times 4)}{4}$$

### Execution Commands
```bash
# Run the fast reduction test
python wine_reduct_simple.py

# Perform data exploration
python explore_uci_wine.py


# example
================================================================================
UCI WINE DATASET - REDUCT TEST (FAST)
================================================================================

Loaded wine dataset: (178, 14)
Classes: [1 2 3]
Class distribution:
1    59
2    71
3    48

Selected 2 samples from classes: [1, 2]

Testing attribute removal:
--------------------------------------------------------------------------------
Attribute                 | FCA Reduct | RST Reduct | Divergence
--------------------------------------------------------------------------------
alcohol                   |     NO     |     YES    |       YES
malic_acid                |     NO     |     NO     |        NO
...

SUMMARY
================================================================================
FCA reducts found: 2
RST reducts found: 5
Divergences (FCA != RST): 3

*** DIVERGENCE FOUND! ***
This demonstrates that with Godel lattice (which fails
double negation), FCA and RST can give different reduct results.
