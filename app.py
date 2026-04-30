from __future__ import annotations

import os

from xp_search.ui import CSS, build_app


if __name__ == "__main__":
    port = int(os.environ.get("XP_APP_PORT", "7860"))
    demo = build_app()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=False,
        show_error=True,
        css=CSS,
    )
