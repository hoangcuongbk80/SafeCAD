"""Small deterministic lifecycle demonstration, no model or dataset downloads."""
import numpy as np
from safecad import SafeCAD, Features


def observation(angle):
    return Features([[np.cos(angle), np.sin(angle)]], [[0.5, 0.5]], 1.0, [1.0])


def main():
    model = SafeCAD()
    model.commission("camera", [observation(0)], [observation(0.02) for _ in range(20)])
    for episode in range(20):
        result = model.observe(observation(0.12), episode, domain="camera")
        if (episode+1) % 4 == 0:
            print(f"window={(episode+1)//4}, pre-update score={result.score:.4f}, "
                  f"pending={len(model.domains['camera'].pending)}, "
                  f"active={len(model.domains['camera'].active)}")
    print(f"Final shifted-normal score: {model.predict(observation(0.12), 'camera').score:.4f}")
    print(f"Unseen-pattern score: {model.predict(observation(0.8), 'camera').score:.4f}")


if __name__ == "__main__":
    main()
