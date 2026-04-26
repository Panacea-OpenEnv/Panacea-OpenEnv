from openenv.core.env_server import create_app

from .panacea_environment import PanaceaEnvironment
from ..models import OversightAction, OversightObservation

_env = PanaceaEnvironment()

app = create_app(lambda: _env, OversightAction, OversightObservation)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
