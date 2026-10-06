"""``python -m grspin`` is equivalent to the ``grspin`` command."""

from .cli import main

raise SystemExit(main())
