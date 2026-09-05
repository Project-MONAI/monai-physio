---
description: Inspect changed MONAI Physio code and existing docstrings, update docstrings and inline comments to match current behavior, and refresh the graphify knowledge graph if public APIs changed.
---

Update documentation for the following in MONAI Physio:

$ARGUMENTS

Instructions:
1. Read the changed source file(s) in full.
2. Read existing docstrings for every public method or class that changed.
3. Update docstrings to reflect current behavior using NumPy docstring style.
   Do not restate ITK shape, axis order, or world space - those are fixed
   project conventions. Document only real deviations from them.
4. Add inline comments only for non-obvious logic (coordinate transforms, shape permutations).
5. Do not create new `.md` files unless explicitly asked.
6. After any public API change - a signature, public behavior, or a public
   attribute - refresh the knowledge graph: `graphify update .`
7. Do not paraphrase the method name as the docstring - explain what it does and why.
