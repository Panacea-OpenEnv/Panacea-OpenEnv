from openenv.core.env_server import create_app

from .panacea_environment import PanaceaEnvironment
from ..models import OversightAction, OversightObservation

app = create_app(PanaceaEnvironment, OversightAction, OversightObservation)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
