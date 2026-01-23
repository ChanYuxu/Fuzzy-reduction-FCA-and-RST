# -*- coding: utf-8 -*-
"""
Simplified Wine Reduct Test - Only First 5 Attributes
"""

import pandas as pd
import sys
sys.path.insert(0, 'f:/FCA_RST')
from fuzzy_fca_reduct import ResiduatedLattice, FuzzyFormalContext, FuzzyRoughContext, is_fca_reduct, is_rst_reduct, get_values_from_data
pd.set_option('display.max_columns', None)

def load_wine():
    column_names = [
        'class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
        'magnesium', 'total_phenols', 'flavanoids',
        'nonflavanoid_phenols', 'proanthocyanins',
        'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline',
        
    ]
    df = pd.read_csv('f:/FCA_RST/wine.data', header=None, names=column_names, sep=',')
    return df

def normalize_to_025(df, cols):
    df_norm = df.copy()
    for col in cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0.5
        df_norm[col] = (df_norm[col] * 4).round() / 4
    return df_norm


def main():
    print("=" * 60)
    print("UCI WINE - REDUCT TEST (First 5 Attributes)")
    print("=" * 60)
    
    df = load_wine()
    print(f"\nOriginal dataset: {df.shape}")
    print(df)

    
    # Select 2 samples from different classes (indices 0 and 59)
    df_samples = df.iloc[[17,61, 68, 132, 153], :]
    # random choose 5 samples
    # df_samples = df.sample(n=5) 
    # check the index of the chosen samples
    print(f"Selected sample indices: {df_samples.index.tolist()}")
    print(f"\nSelected samples: {df_samples.shape}")
    print(f"  Class 0: {df_samples.iloc[0]['class']}")
    print(f"  Class 1: {df_samples.iloc[1]['class']}")
    print(f"  Class 2: {df_samples.iloc[2]['class']}")
    print(f"  Class 3: {df_samples.iloc[3]['class']}")
    print(f"  Class 4: {df_samples.iloc[4]['class']}")

    
    # Use first 5 attributes only
    attrs = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium' ,'total_phenols','flavanoids']# , 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']#'total_phenols', ,
    df_test = df_samples[attrs].copy()
    print(df_test)
    
    # Normalize to 0.25 precision
    df_test = normalize_to_025(df_test, attrs)
    
    print(f"\nTest data (normalized, 0.25 precision):")
    print(df_test.round(2))
    
    for operator in ['godel','lukasiewicz']:
        print(f'Now calculate under {operator}')
        # Create contexts
        fca = FuzzyFormalContext(df_test, ResiduatedLattice(operator))
        rst = FuzzyRoughContext(df_test, ResiduatedLattice(operator))
        # do the negation operator on the relation
        df_test_neg = df_test.applymap(lambda x: ResiduatedLattice(operator).neg(x))
        rst_neg = FuzzyRoughContext(df_test_neg, ResiduatedLattice(operator))

        # Get fuzzy values from data (matches 0.25 precision)
        fuzzy_values = get_values_from_data(df_test)
        print(f"\nFuzzy values in data: {fuzzy_values}")
        print(f"Will test with {len(fuzzy_values)**len(df_test)} subsets (for {len(df_test)} objects)")

        print("\n" + "=" * 60)
        print("TESTING ATTRIBUTE REMOVAL")
        print("=" * 60)

        attributes = list(df_test.columns)
        divergences = []
        divergences_neg = []

        for attr in attributes:
            X_prime = set(df_test.index)
            Y_prime = set(attributes) - {attr}

            try:
                fca_ok, fca_msg = is_fca_reduct(fca, X_prime, Y_prime, False, fuzzy_values)
                rst_ok, rst_msg = is_rst_reduct(rst, X_prime, Y_prime, False, fuzzy_values)
                rst_neg_ok, rst_neg_msg = is_rst_reduct(rst_neg, X_prime, Y_prime, False, fuzzy_values)
            except Exception as e:
                print(f"\nError testing '{attr}': {e}")
                continue
            
            fca_mark = "YES" if fca_ok else "NO"
            rst_mark = "YES" if rst_ok else "NO"
            rst_neg_mark = "YES" if rst_neg_ok else "NO"
            div = " [DIVERGENCE!]" if fca_ok != rst_ok else ""
            
            if fca_ok != rst_ok:
                divergences.append(attr)
            if fca_ok != rst_neg_ok:
                divergences_neg.append(attr)
            
            print(f"Remove '{attr:20s}': FCA={fca_mark:3s} | RST={rst_mark:3s} | NEG_RST={rst_neg_mark:3s}")
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if divergences:
            print(f"Divergences found: {len(divergences)}")
            for d in divergences:
                print(f"  - {d}")
        if divergences_neg:
            print(f"Divergences found: {len(divergences_neg)}")
            for d in divergences_neg:
                print(f"  - {d}")
        else:
            print("No divergence found in tested subsets.")
        
        print('=' * 60)


if __name__ == '__main__':
    # for i in range(20):
    #     print(f"testing {i}-th random search")
    main()
