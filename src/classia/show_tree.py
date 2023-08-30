import hier
import os

# you can see some tree structures like this. For index, check generated labels.txt.
# 0
# ├── 1
# │   ├── 2
# │   │   ├── 3
# │   │   ├── 4

model_name = 'amazon_product_reviews'
tree_path = os.path.expanduser(f"~/.cache/classia/models/{model_name}/tree.csv")

with open(tree_path) as f:
    tree, tree_names = hier.make_hierarchy_from_edges(hier.load_edges(f))

    print(tree)
