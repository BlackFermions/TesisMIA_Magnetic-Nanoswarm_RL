"""Integración Euler–Maruyama vectorizada con paredes reflectantes.

Física del entorno v0.1 (ver docs/diseno_entorno.md): bifurcación en Y en 2D,
flujo de Poiseuille analítico por tramo y N partículas no interactuantes.
Todo está en unidades adimensionales (W = 1, U_max = 1).
"""

from dataclasses import dataclass

import numpy as np

# Índices de tramo
PARENT, TARGET, OTHER = 0, 1, 2

# Estados de cada partícula
ACTIVE, AT_TARGET, AT_OTHER = 0, 1, 2


@dataclass(frozen=True)
class Condition:
    """Nivel de fidelidad física."""

    name: str
    flow: str  # "none", "steady" o "pulsatile"
    brownian: bool


CONDITIONS = {
    "R1": Condition("R1", flow="none", brownian=False),
    "R2": Condition("R2", flow="steady", brownian=False),
    "R3": Condition("R3", flow="pulsatile", brownian=False),
    "R4": Condition("R4", flow="pulsatile", brownian=True),
}


@dataclass(frozen=True)
class PhysicsParams:
    u_max: float = 1.0  # velocidad en el centro del canal
    pulse_amplitude: float = 0.5  # A en U(t) = U_max (1 + A sin(2πt/T))
    pulse_period: float = 4.0  # T
    v_mag: float = 0.3  # velocidad magnética máxima relativa a U_max
    diffusion: float = 0.01  # D, solo se usa si la condición tiene browniano
    dt: float = 0.05


class YBifurcation:
    """Canal madre horizontal que se divide en dos ramas a ±theta.

    La rama superior es la rama objetivo. Cada tramo se describe por su línea
    central: origen, vector unitario tangente, normal (tangente girada +90°) y
    largo. Una partícula pertenece al tramo cuya línea central tiene más cerca.
    """

    def __init__(self, width=1.0, parent_length=5.0, branch_length=4.0, theta_deg=30.0):
        self.width = width
        self.half_width = width / 2
        self.parent_length = parent_length
        self.branch_length = branch_length
        self.theta = np.deg2rad(theta_deg)

        junction = np.array([parent_length, 0.0])
        c, s = np.cos(self.theta), np.sin(self.theta)
        self.junction = junction
        self.origins = np.array([[0.0, 0.0], junction, junction])
        self.tangents = np.array([[1.0, 0.0], [c, s], [c, -s]])
        self.normals = np.stack([-self.tangents[:, 1], self.tangents[:, 0]], axis=1)
        self.lengths = np.array([parent_length, branch_length, branch_length])

    def project(self, pos):
        """Asigna cada punto a su tramo más cercano.

        Devuelve (tramo, t, s): t es la coordenada a lo largo de la línea
        central y s la distancia lateral con signo.
        """
        rel = pos[:, None, :] - self.origins[None, :, :]  # (n, 3, 2)
        t = np.einsum("nkd,kd->nk", rel, self.tangents)
        s = np.einsum("nkd,kd->nk", rel, self.normals)
        t_clip = np.clip(t, 0.0, self.lengths)
        dist = np.hypot(t - t_clip, s)  # distancia al segmento
        seg = np.argmin(dist, axis=1)
        idx = np.arange(len(pos))
        return seg, t[idx, seg], s[idx, seg]

    def to_xy(self, seg, t, s):
        return self.origins[seg] + t[:, None] * self.tangents[seg] + s[:, None] * self.normals[seg]

    def inside(self, seg, t, s):
        ok = np.abs(s) <= self.half_width
        return ok & ~((seg == PARENT) & (t < 0))

    def distance_to_target(self, seg, t):
        """Distancia vascular (sobre la línea central) hasta la salida objetivo."""
        L0, L1 = self.parent_length, self.branch_length
        d_parent = (L0 - np.clip(t, 0, L0)) + L1
        d_target = L1 - np.clip(t, 0, L1)
        d_other = np.clip(t, 0, L1) + L1  # volver a la bifurcación y subir
        return np.select([seg == PARENT, seg == TARGET], [d_parent, d_target], d_other)

    def initial_positions(self, n, rng):
        """Enjambre liberado en la entrada, repartido a lo ancho del canal."""
        t = rng.uniform(0.0, 0.2 * self.width, n)
        s = rng.uniform(-0.45 * self.width, 0.45 * self.width, n)
        return self.to_xy(np.zeros(n, dtype=int), t, s)


def flow_speed(condition, params, time):
    """Velocidad en el centro del canal en el instante `time`."""
    if condition.flow == "none":
        return 0.0
    if condition.flow == "steady":
        return params.u_max
    phase = 2 * np.pi * time / params.pulse_period
    return params.u_max * (1 + params.pulse_amplitude * np.sin(phase))


def flow_velocity(geom, seg, s, u_center):
    """Perfil de Poiseuille u(s) = U (1 - (2s/W)^2) en la dirección del tramo."""
    profile = np.clip(1 - (s / geom.half_width) ** 2, 0.0, None)
    return (u_center * profile)[:, None] * geom.tangents[seg]


def step(geom, condition, params, pos, status, force, time, rng):
    """Avanza un paso Δt las partículas activas.

    pos: (n, 2) posiciones; status: (n,) estados; force: (2,) acción con norma ≤ 1.
    Devuelve las nuevas (pos, status). Las partículas entregadas no se mueven.
    """
    active = status == ACTIVE
    if not active.any():
        return pos, status

    p = pos[active]
    seg, _, s = geom.project(p)
    velocity = flow_velocity(geom, seg, s, flow_speed(condition, params, time))
    velocity = velocity + params.v_mag * force

    new = p + velocity * params.dt
    if condition.brownian:
        new = new + np.sqrt(2 * params.diffusion * params.dt) * rng.standard_normal(new.shape)

    # Paredes reflectantes: se refleja la componente lateral; en la entrada,
    # la longitudinal. Si aun así queda fuera, la partícula no se mueve.
    seg, t, s = geom.project(new)
    out = np.abs(s) > geom.half_width
    s = np.where(out, np.sign(s) * (geom.width - np.abs(s)), s)
    t = np.where((seg == PARENT) & (t < 0), -t, t)
    new = geom.to_xy(seg, t, s)
    seg, t, s = geom.project(new)
    bad = ~geom.inside(seg, t, s)
    new[bad] = p[bad]
    seg, t, s = geom.project(new)

    # Entrega: la partícula cruza el extremo de una rama
    new_status = np.full(len(p), ACTIVE)
    exited = (seg != PARENT) & (t >= geom.branch_length)
    new_status[exited & (seg == TARGET)] = AT_TARGET
    new_status[exited & (seg == OTHER)] = AT_OTHER

    pos = pos.copy()
    status = status.copy()
    pos[active] = new
    status[active] = new_status
    return pos, status
