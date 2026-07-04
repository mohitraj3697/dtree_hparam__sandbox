import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, plot_tree
from mlxtend.plotting import plot_decision_regions

st.set_page_config(
    page_title="Decision Tree Playground",
    page_icon="DT",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-text {
        font-size: 1.05rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #30475e;
    }
    .stButton > button {
        background: linear-gradient(90deg, #e94560, #0f3460);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.85;
    }
    div[data-testid="stSidebar"] {
        background: #0a0a1a;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Decision Tree Hyperparameter Sandbox</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Drag the sliders, hit run, and watch how each knob reshapes the tree and its decision boundary.</p>', unsafe_allow_html=True)

st.sidebar.header("Hyperparameters")
st.sidebar.caption("Tweak these and see what happens")

criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"], help="Split quality measure — gini impurity or entropy (information gain)")
splitter = st.sidebar.selectbox("Splitter", ["best", "random"], help="'best' picks the optimal split; 'random' adds noise to reduce overfitting")

use_max_depth = st.sidebar.checkbox("Limit max depth", value=True)
max_depth = None
if use_max_depth:
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5, help="How deep the tree can grow. None = grows until every leaf is pure (usually overfits)")

min_samples_split = st.sidebar.slider("Min Samples Split", 2, 50, 2, help="Minimum samples a node needs before it can split further")
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 50, 1, help="Minimum samples required in each leaf after a split")

max_features_option = st.sidebar.selectbox("Max Features", ["None", "1", "2"], help="Features considered at each split. Only 2 available in this toy dataset")
max_features = None if max_features_option == "None" else int(max_features_option)

use_max_leaf = st.sidebar.checkbox("Limit max leaf nodes", value=False)
max_leaf_nodes = None
if use_max_leaf:
    max_leaf_nodes = st.sidebar.slider("Max Leaf Nodes", 2, 50, 10, help="Caps total leaves — another way to control complexity")

min_impurity_decrease = st.sidebar.slider("Min Impurity Decrease", 0.0, 0.5, 0.0, step=0.01, help="Split only if impurity drops by at least this much")

st.sidebar.markdown("---")
st.sidebar.caption("Dataset settings")
n_samples = st.sidebar.slider("Number of samples", 100, 1000, 300, step=50)
noise_level = st.sidebar.slider("Noise", 0.05, 0.5, 0.2, step=0.05)

run = st.sidebar.button("Run Algorithm")

if run:
    X, y = make_moons(n_samples=n_samples, noise=noise_level, random_state=42)

    clf = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        random_state=42
    )
    clf.fit(X, y)

    train_acc = clf.score(X, y)

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Training Accuracy", f"{train_acc:.4f}")
    with col_info2:
        st.metric("Tree Depth", clf.get_depth())
    with col_info3:
        st.metric("Number of Leaves", clf.get_n_leaves())

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Decision Boundary")
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        plot_decision_regions(X, y.astype(np.int64), clf=clf, ax=ax1, legend=2)
        ax1.set_xlabel("Feature 1")
        ax1.set_ylabel("Feature 2")
        ax1.set_title("How the tree carves up the space")
        fig1.tight_layout()
        st.pyplot(fig1)

    with col2:
        st.subheader("Tree Structure")
        depth = clf.get_depth()
        fig_h = max(6, depth * 1.5)
        fig_w = max(12, clf.get_n_leaves() * 1.2)
        fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
        plot_tree(
            clf,
            filled=True,
            rounded=True,
            ax=ax2,
            feature_names=["Feature 1", "Feature 2"],
            class_names=["Class 0", "Class 1"],
            fontsize=9,
            impurity=True
        )
        ax2.set_title("The actual tree sklearn built")
        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")
    with st.expander("Quick notes on what you're seeing"):
        st.markdown("""
        - **Accuracy shown is training accuracy** — the model is predicting on the same data it learned from. 
          A really deep tree will easily hit ~1.0 here, but that doesn't mean it generalizes well.
        - **Decision boundary** shows how the tree partitions the 2D feature space. 
          More complex trees = more jagged, fragmented regions.
        - **Tree diagram** shows the actual splits. Each node shows the feature/threshold it splits on, 
          the impurity (gini/entropy), and how many samples ended up there.
        - Try cranking `max_depth` to 20 with no other constraints — you'll see classic overfitting in action.
        """)

else:
    st.info("Set your hyperparameters in the sidebar and hit **Run Algorithm** to see results.")

    with st.expander("What are all these parameters?", expanded=True):
        st.markdown("""
        | Parameter | What it controls |
        |---|---|
        | **Criterion** | How the tree measures split quality — `gini` impurity or `entropy` |
        | **Splitter** | `best` tries every split; `random` picks one at random (can reduce overfitting) |
        | **Max Depth** | How many levels deep the tree can go |
        | **Min Samples Split** | Minimum samples needed in a node before it's allowed to split |
        | **Min Samples Leaf** | Minimum samples that must end up in each leaf |
        | **Max Features** | How many features to consider per split |
        | **Max Leaf Nodes** | Hard cap on total leaves |
        | **Min Impurity Decrease** | Only split if the impurity improvement is above this threshold |
        """)
