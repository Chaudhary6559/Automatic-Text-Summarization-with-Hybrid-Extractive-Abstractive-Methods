import sys
from pathlib import Path

import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.src.config import get_settings


def run() -> None:
    # Get config path relative to backend directory
    config_path = Path(__file__).parent / "config" / "config.yaml"
    if config_path.exists():
        settings = get_settings(config_path)
    else:
        # Fallback to default settings
        settings = get_settings()
    
    uvicorn.run(
        "backend.app:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.reload,
        log_level=settings.app.log_level,
    )


if __name__ == "__main__":
    run()


