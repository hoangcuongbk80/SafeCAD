"""Causal Quarantine-and-Consolidate Memory for SafeCAD.

No labels, masks, adaptive scores or searchable references enter eligibility.
All memory updates use deterministic exact feature search.
"""
from dataclasses import dataclass, field
import math
from typing import Callable
import numpy as np
from .config import Config
from .features import Features, normalize, distances, nearest, top_mean, retained_indices, coreset_indices


@dataclass
class Observation:
    arrival: int
    episode: int
    window: int
    g: np.ndarray
    z: np.ndarray
    coords: np.ndarray
    patch_indices: np.ndarray
    cg: float
    cp: np.ndarray
    anchor_distances: np.ndarray
    consolidated: np.ndarray


@dataclass
class Prototype:
    arrival: int
    patch_index: int
    episode: int
    origin_window: int
    created: int
    z: np.ndarray
    coord: np.ndarray
    g: np.ndarray
    episodes: set = field(default_factory=set)
    windows: set = field(default_factory=set)
    episode_count: int = 0
    window_count: int = 0

    @property
    def key(self):
        return self.arrival, self.patch_index


@dataclass
class Domain:
    anchor: np.ndarray
    initial_context: np.ndarray
    adaptive_context: np.ndarray
    threshold: float
    pending: list = field(default_factory=list)
    active: list = field(default_factory=list)
    quarantine: list = field(default_factory=list)
    completed_episodes: int = 0
    completed_window: int = -1


@dataclass
class Prediction:
    domain: str | None
    nearest_domain: str | None
    abstained: bool
    calibrated: bool
    patch_scores: np.ndarray | None
    score: float | None
    threshold: float | None
    anomalous: bool | None
    anchor_scores: np.ndarray | None = None


class SafeCAD:
    def __init__(self, config: Config | None = None, event_sink: Callable | None = None):
        self.config = config or Config()
        self.domains = {}
        self.unsupported = []
        self.unsupported_completed = 0
        self.arrival = 0
        self.last_episode = -1
        self.open_episode = None
        self.open_domain = None
        self.open_nearest = None
        self.open_frame = 0
        self.open_retained = 0
        self.event_sink = event_sink
        self.last_boundary = None

    def _emit(self, kind, domain, **data):
        if self.event_sink:
            self.event_sink(dict(event=kind, domain=domain, **data))

    def commission(self, domain: str, initialization, calibration):
        """Verified normal, disjoint ordered lists. Calibration never enters memory."""
        c = self.config
        if not isinstance(domain, str) or not domain:
            raise ValueError("domain must be a nonempty string")
        if self.open_episode is not None:
            raise RuntimeError("finish the open episode before commissioning")
        if domain in self.domains:
            raise ValueError("domain already commissioned")
        if len(self.domains) >= c.max_domains:
            raise ValueError("maximum commissioned domains reached")
        initialization, calibration = list(initialization), list(calibration)
        if not initialization or len(calibration) < c.calibration_min:
            raise ValueError(f"need initialization and at least {c.calibration_min} calibration observations")
        if {id(x) for x in initialization} & {id(x) for x in calibration}:
            raise ValueError("initialization and calibration must be disjoint")
        dim = initialization[0].z.shape[1]
        if any(x.z.shape[1] != dim for x in initialization + calibration):
            raise ValueError("feature dimensions differ")
        if self.domains and next(iter(self.domains.values())).anchor.shape[1] != dim:
            raise ValueError("all domains must share one representation")
        all_z = np.concatenate([x.z for x in initialization])
        anchor = all_z[coreset_indices(all_z, c.anchor_capacity)].copy()
        mu = normalize(np.mean([x.descriptor(c.trim_fraction) for x in initialization], axis=0))
        scores = sorted(top_mean(nearest(x.z, anchor, c.distance_chunk), c.top_fraction) for x in calibration)
        threshold = scores[math.ceil((1-c.alpha)*len(scores))-1]
        anchor.setflags(write=False)
        mu.setflags(write=False)
        self.domains[domain] = Domain(anchor, mu, mu.copy(), threshold)

    def route(self, g):
        if not self.domains:
            return None, None
        # Commissioning order resolves equal domain distances deterministically.
        name = min(self.domains, key=lambda d: min(
            np.linalg.norm(g-self.domains[d].initial_context),
            np.linalg.norm(g-self.domains[d].adaptive_context)))
        state = self.domains[name]
        distance = min(np.linalg.norm(g-state.initial_context), np.linalg.norm(g-state.adaptive_context))
        return (name if distance <= self.config.route_radius else None), name

    def _predict(self, x, domain, nearest_domain):
        c = self.config
        name = domain if domain is not None else nearest_domain
        if name is None:
            return Prediction(None, None, True, False, None, None, None, None)
        state = self.domains[name]
        if x.z.shape[1] != state.anchor.shape[1]:
            raise ValueError("feature dimension differs from commissioned encoder")
        a = nearest(x.z, state.anchor, c.distance_chunk)
        scores = a.copy()
        if domain is not None and state.active:
            scores = np.minimum(scores, nearest(x.z, np.stack([p.z for p in state.active]), c.distance_chunk))
        score = top_mean(scores, c.top_fraction)
        supported = domain is not None
        return Prediction(domain, nearest_domain, not supported, supported, scores, score,
                          state.threshold if supported else None,
                          score > state.threshold if supported else None, a)

    def predict(self, x: Features, domain: str | None = None):
        """Read-only inference, suitable for frozen-checkpoint evaluation."""
        if domain is not None:
            if domain not in self.domains:
                return self._predict(x, None, None)
            return self._predict(x, domain, domain)
        selected, near = self.route(x.descriptor(self.config.trim_fraction))
        return self._predict(x, selected, near)

    def needs_consistency(self):
        """Whether the next frame will be retained and needs transformed views."""
        c = self.config
        return self.open_episode is None or (
            self.open_frame % c.retention_stride == 0 and self.open_retained < c.episode_cap)

    def observe(self, x: Features, episode: int, domain: str | None = None, end_episode=True):
        """Score before storing; explicit episode completion drives window boundaries.

        Episodes are contiguous and use strictly increasing nonnegative integer
        acquisition IDs. Set end_episode=False for all but the final frame.
        Task-free routing is locked by the first frame of each episode.
        """
        c = self.config
        if not isinstance(episode, int) or isinstance(episode, bool) or episode < 0:
            raise ValueError("episode must be a nonnegative integer")
        if self.needs_consistency() and (x.global_consistency is None or x.patch_consistency is None):
            raise ValueError("adaptation requires measured photometric consistency")
        if self.domains and x.z.shape[1] != next(iter(self.domains.values())).anchor.shape[1]:
            raise ValueError("feature dimension differs from commissioned encoder")
        g = x.descriptor(c.trim_fraction)
        if self.open_episode is None:
            if episode <= self.last_episode:
                raise ValueError("episode IDs must increase and cannot be revisited")
            selected, near = self.route(g) if domain is None else (
                (domain, domain) if domain in self.domains else (None, None))
            self.open_episode, self.open_domain, self.open_nearest = episode, selected, near
            self.open_frame = self.open_retained = 0
        elif episode != self.open_episode:
            raise ValueError("complete the current episode before starting another")
        elif domain is not None and domain != self.open_domain:
            # An unknown supplied domain is unsupported for its entire episode.
            if self.open_domain is not None:
                raise ValueError("domain assignment cannot change within an episode")
        prediction = self._predict(x, self.open_domain, self.open_nearest)
        self.arrival += 1
        if self.open_frame % c.retention_stride == 0 and self.open_retained < c.episode_cap:
            ids = retained_indices(x.coords, c.layout_rows, c.layout_cols)
            state = self.domains.get(self.open_domain)
            window = (state.completed_episodes if state else self.unsupported_completed) // c.episodes_per_window
            anchor = prediction.anchor_scores[ids].copy() if prediction.anchor_scores is not None else np.full(len(ids), np.inf, np.float32)
            obs = Observation(self.arrival, episode, window, g.copy(), x.z[ids].copy(),
                              x.coords[ids].copy(), ids, float(x.global_consistency),
                              x.patch_consistency[ids].copy(), anchor, np.zeros(len(ids), dtype=bool))
            target = state.quarantine if state else self.unsupported
            target.append(obs)
            capacity = c.quarantine_capacity if state else c.unsupported_capacity
            while len(target) > capacity:
                old = target.pop(0)
                self._emit("quarantine_evicted", self.open_domain, arrival=old.arrival)
            self.open_retained += 1
            self._emit("retained", self.open_domain, arrival=obs.arrival, episode=episode,
                       window=window, patch_indices=ids.tolist())
        self.open_frame += 1
        if end_episode:
            self.finish_episode()
        return prediction

    def finish_episode(self):
        """Complete an open episode without requiring an additional observation."""
        if self.open_episode is None:
            raise RuntimeError("no open episode")
        c = self.config
        name = self.open_domain
        if name is not None:
            state = self.domains[name]
            state.completed_episodes += 1
            if state.completed_episodes % c.episodes_per_window == 0:
                self._boundary(name, state.completed_episodes // c.episodes_per_window - 1)
        else:
            self.unsupported_completed += 1
            if self.unsupported_completed % c.episodes_per_window == 0:
                b = self.unsupported_completed // c.episodes_per_window - 1
                self.unsupported = [o for o in self.unsupported if b-o.window < c.unsupported_lifetime]
        self.last_episode = self.open_episode
        self.open_episode = self.open_domain = self.open_nearest = None

    def _eligible(self, observations):
        """Compute all gates on one frozen completed-window snapshot."""
        c = self.config
        eligible = {}
        for o in observations:
            other_episodes = {k.episode for k in observations if k.episode != o.episode}
            peers = [k for k in observations if k.episode != o.episode
                     and k.cg >= c.context_consistency
                     and np.linalg.norm(o.g-k.g) <= c.context_radius]
            episodes = {k.episode for k in peers}
            windows = {k.window for k in peers}
            cg_ok = (o.cg >= c.context_consistency and len(episodes) >= c.context_episodes
                     and len(windows) >= c.context_windows)
            good = np.zeros(len(o.z), dtype=bool)
            if cg_ok:
                patch_episodes = [set() for _ in o.z]
                patch_windows = [set() for _ in o.z]
                for k in peers:
                    match = ((distances(o.z, k.z) <= c.patch_radius)
                             & (distances(o.coords, k.coords) <= c.spatial_radius)
                             & (k.cp[None, :] >= c.patch_consistency))
                    for j in np.flatnonzero(match.any(axis=1)):
                        patch_episodes[j].add(k.episode)
                        patch_windows[j].add(k.window)
                good = np.array([o.cp[j] >= c.patch_consistency
                    and len(patch_episodes[j]) >= c.patch_episodes
                    and len(patch_windows[j]) >= c.patch_windows for j in range(len(o.z))])
            eligible[o.arrival] = good
            self._emit("context_support", None, arrival=o.arrival, episodes=len(episodes),
                       windows=len(windows), fraction=len(episodes)/max(1, len(other_episodes)))
        return eligible

    def _select(self, entries, capacity):
        if not entries:
            return []
        ids = coreset_indices(np.stack([p.z for p in entries]), capacity, [p.key for p in entries])
        return [entries[i] for i in ids]

    def _matches(self, p, o, patch_ids, radius):
        c = self.config
        return (np.linalg.norm(p.g-o.g) <= c.context_radius and len(patch_ids) > 0
                and np.any((np.linalg.norm(o.z[patch_ids]-p.z, axis=1) <= radius)
                           & (np.linalg.norm(o.coords[patch_ids]-p.coord, axis=1) <= c.spatial_radius)))

    def _duplicate(self, p, references):
        c = self.config
        return any(np.linalg.norm(p.z-r.z) <= c.duplicate_radius
                   and np.linalg.norm(p.g-r.g) <= c.context_radius
                   and np.linalg.norm(p.coord-r.coord) <= c.spatial_radius for r in references)

    def _boundary(self, name, b):
        c, state = self.config, self.domains[name]
        expired = [o for o in state.quarantine if b-o.window >= c.quarantine_lifetime]
        for o in expired:
            self._emit("quarantine_expired", name, arrival=o.arrival, window=b)
        state.quarantine = [o for o in state.quarantine if b-o.window < c.quarantine_lifetime]
        snapshot = [o for o in state.quarantine if o.window <= b]
        eligible = self._eligible(snapshot)
        activated, surviving = [], []
        for p in state.pending:
            # Only observations from the window just completed can vote now.
            for o in snapshot:
                if o.window == b and b > p.created and self._matches(p, o, np.flatnonzero(eligible[o.arrival]), c.patch_radius):
                    if p.episode_count < c.activation_episodes and o.episode not in p.episodes:
                        p.episodes.add(o.episode)
                        p.episode_count += 1
                    if p.window_count < c.activation_windows and b not in p.windows:
                        p.windows.add(b)
                        p.window_count += 1
            if p.episode_count >= c.activation_episodes and p.window_count >= c.activation_windows:
                p.episodes.clear()
                p.windows.clear()
                activated.append(p)
            elif b-p.created >= c.pending_lifetime:
                self._emit("pending_expired", name, origin=list(p.key), window=b)
            else:
                surviving.append(p)
        old_active = state.active
        state.active = self._select(old_active + activated, c.active_capacity)
        active_keys = {p.key for p in state.active}
        newly_retained = [p for p in activated if p.key in active_keys]
        for p in activated:
            self._emit("activated" if p.key in active_keys else "activation_rejected", name,
                       origin=list(p.key), episode=p.episode, window=b)
        for p in old_active:
            if p.key not in active_keys:
                self._emit("active_evicted", name, origin=list(p.key), window=b)
        candidates, origins = [], {}
        references = state.active + surviving
        for o in sorted(snapshot, key=lambda k: k.arrival):
            for j in sorted(np.flatnonzero(eligible[o.arrival] & ~o.consolidated), key=lambda j: o.patch_indices[j]):
                p = Prototype(o.arrival, int(o.patch_indices[j]), o.episode, o.window, b,
                              o.z[j].copy(), o.coords[j].copy(), o.g.copy())
                if not self._duplicate(p, references + candidates):
                    candidates.append(p)
                    origins[p.key] = (o, j)
        state.pending = self._select(surviving + candidates, c.pending_capacity)
        pending_keys = {p.key for p in state.pending}
        for p in surviving + candidates:
            if p.key not in pending_keys:
                self._emit("pending_coreset_rejected", name, origin=list(p.key), window=b)
        for p in candidates:
            if p.key in pending_keys:
                o, j = origins[p.key]
                o.consolidated[j] = True
                self._emit("pending_admitted", name, origin=list(p.key), episode=p.episode, window=b)
        if newly_retained:
            origins_g = {p.arrival: p.g for p in newly_retained}
            mean_g = normalize(np.stack(list(origins_g.values())).mean(axis=0))
            state.adaptive_context = normalize((1-c.context_eta)*state.adaptive_context + c.context_eta*mean_g)
        state.completed_window = b
        self.last_boundary = dict(domain=name, window=b, quarantine=len(state.quarantine),
            pending=len(state.pending), active=len(state.active), activated=len(newly_retained))
        self._emit("boundary", name, **{k:v for k,v in self.last_boundary.items() if k != "domain"})

    def rollback(self, domain):
        """Explicitly reset one domain to trusted anchors and original context."""
        if self.open_episode is not None:
            raise RuntimeError("finish the open episode before rollback")
        state = self.domains[domain]
        state.pending.clear()
        state.active.clear()
        state.quarantine.clear()
        state.adaptive_context = state.initial_context.copy()
        self._emit("rollback", domain)

    def save(self, path):
        from .checkpoint import save
        save(self, path)

    @classmethod
    def load(cls, path, event_sink=None):
        from .checkpoint import load
        return load(path, event_sink)
