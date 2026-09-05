---
description: Inspect MONAI Physio source files, summarize the current design, and produce a numbered implementation plan with open questions. Does not write code unless explicitly asked.
---

Analyze the following and produce a design plan for the MONAI Physio repository.

Task: $ARGUMENTS

Instructions:
1. Use `graphify query "<question>"` to locate relevant classes and methods, then read those source files.
2. Summarize current behavior in 3–5 bullet points.
3. Produce a numbered implementation plan with enough detail to act on.
4. List every file that will change.
5. Call out any change that breaks the fixed data conventions - an axis
   permutation, an ITK-to-NumPy handoff, or an LPS-to-USD conversion. Do not
   restate the conventions themselves.
6. List open questions that need user input before coding starts.
7. Do not modify any files unless the task explicitly asks you to.
