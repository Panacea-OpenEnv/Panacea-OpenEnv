"""Root entry point. Re-exports the FastAPI app for the OpenEnv `server` script."""

import os
import uvicorn

from openenv_panacea.server.app import app

__all__ = ["app", "main"]


def main():
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
