=============
AI Assistants
=============

MONAI Physio is developed with AI coding assistants in the loop (Claude Code,
Codex, Cursor, and others). Two repository conventions make that work well.

Repository Instructions
=======================

``AGENTS.md`` holds the role-based guidance that applies across every
assistant; ``CLAUDE.md`` adds Claude-specific instructions. Role subagents live
in ``.agents/agents/`` and slash-command skills in ``.agents/skills/``. Read
those before hand-writing a prompt that repeats project conventions.

One of those conventions binds every assistant: MONAI Physio prefers
compatibility, breaks a public API only when the change is generally beneficial
to future users, and never ships deprecation shims. Any commit that does break
an API must add an entry to :doc:`migration_next` describing the change and the
code that automates the conversion.

graphify
========

The recommended way to explore this codebase with an AI assistant is the
graphify knowledge graph in ``graphify-out/``, which indexes god nodes,
community structure, and cross-file relationships.

A graph query returns a scoped subgraph. That is dramatically smaller and more
accurate than pasting whole files or raw ``grep`` output into a prompt, which
matters here: the source tree spans segmentation, registration, statistical
modeling, and USD export, and the relevant call chain for a question is rarely
in one file.

.. code-block:: bash

   graphify query "how does a labelmap become a USD stage?"
   graphify path "WorkflowConvertImageToVTK" "ConvertVTKToUSD"
   graphify explain "anatomy taxonomy"

Recommended use:

* Ask ``graphify query`` before searching manually, whenever
  ``graphify-out/graph.json`` exists.
* Use ``graphify path`` to see how two classes relate, and ``graphify explain``
  to focus on a single concept.
* Browse ``graphify-out/wiki/index.md`` for broad navigation; read
  ``graphify-out/GRAPH_REPORT.md`` only for whole-architecture review.
* Run ``graphify update .`` after changing code. The update is AST-only, so it
  costs no API calls, and it keeps assistants from resolving symbols against a
  stale graph.

See Also
========

* :doc:`core`
* :doc:`extending`
* :doc:`migration_next`
* :doc:`../contributing`
