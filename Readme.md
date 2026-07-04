## dtree-hparam-sandbox
 
A Streamlit app for tuning Decision Tree hyperparameters and watching how they reshape both the decision boundary and the actual tree structure, side by side.

Same idea as my logistic regression sandbox — decision trees have way more knobs than logreg, and it's hard to build intuition for `min_samples_leaf` vs `max_leaf_nodes` vs `min_impurity_decrease` just from docs. Seeing the tree literally grow or shrink as you drag a slider makes it way easier to reason about overfitting.
 
## What it does
 
- Generates a non-linear `make_moons` toy dataset (so the tree actually has to work for it, unlike a linearly separable blob)
- Lets you tune every major Decision Tree hyperparameter from the sidebar:
  - `criterion` — gini or entropy
  - `splitter` — best or random
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
  - `max_features`
  - `max_leaf_nodes`
  - `min_impurity_decrease`
- Hit "Run Algorithm" and it fits the tree, then shows:
  - the decision boundary (via `mlxtend`'s `plot_decision_regions`)
  - the actual tree structure (via sklearn's `plot_tree`)
  side by side, plus training accuracy

## Parameter reference
 
- **Criterion** — the split quality measure, either `gini` impurity or `entropy` (information gain)
- **Splitter** — `best` evaluates every possible split and picks the optimal one; `random` picks a random split, which can help reduce overfitting
- **Max Depth** — caps how deep the tree can grow. Leave it as `None` and the tree grows until every leaf is pure, which almost always overfits. Setting a number forces simpler, more general rules
- **Min Samples Split** — the minimum samples a node needs before it's allowed to split further. Raising this stops the tree from splitting on tiny, noisy subsets
- **Min Samples Leaf** — minimum samples required in each leaf after a split. A split only happens if both resulting branches meet this minimum, which smooths out the tree and kills off tiny leaves
- **Max Features** — how many features are considered at each split. Restricting this adds randomness and helps in higher-dimensional datasets (less relevant here since we only have 2 features, but useful to see how it behaves)
- **Max Leaf Nodes** — caps the total number of leaves, growing the tree best-first up to that limit. Another way to control complexity besides depth
- **Min Impurity Decrease** — a split only happens if it improves purity by at least this much. Push this up and the tree stops splitting unless the gain is actually worth it

## Running it locally
 
```bash
git clone https://github.com/mohitraj3697/dtree_hparam_sandbox.git
cd dtree_hparam_sandbox
pip install -r requirements.txt
streamlit run main.py
```
 
Opens at `localhost:8501`.


## Notes
 
- Accuracy shown is training accuracy (predicting on the same `X` it was fit on), not a held-out test set — good enough for seeing overfitting happen visually (a super deep tree will hit ~1.0 easily), but not a real evaluation
- `max_features` slider only goes up to 2 since the toy dataset only has 2 columns — doesn't reflect how useful this param is on real, higher-dimensional data
- No dataset upload yet, just the fixed moons dataset

## Why this exists
 
Wanted to actually see how each hyperparameter trades off bias and variance instead of just tuning blind with GridSearchCV. Watching the tree get simpler or more tangled as you move a slider makes overfitting a lot more intuitive than reading about it.
 
## License
 
MIT