"""Dimensionnement des effectifs par la thEorie des files d'attente (Erlang C et Erlang A).

Traduit un volume d'appels prEvu et un AHT en nombre d'agents requis pour atteindre un
objectif de niveau de service. Erlang C ignore l'abandon (borne haute) ; une approximation
d'Erlang A (clients impatients) corrige A la baisse le besoin en agents.
"""
from __future__ import annotations
import math


def erlang_c_prob_wait(n_agents: int, traffic_a: float) -> float:
    """ProbabilitE qu'un appel attende (formule d'Erlang C)."""
    if n_agents <= traffic_a:
        return 1.0
    # Somme des termes de Poisson
    s = sum((traffic_a ** k) / math.factorial(k) for k in range(n_agents))
    last = (traffic_a ** n_agents) / math.factorial(n_agents) * (n_agents / (n_agents - traffic_a))
    return last / (s + last)


def service_level(n_agents: int, traffic_a: float, aht_s: float, target_s: float) -> float:
    """Niveau de service Erlang C : P(attente <= target_s)."""
    if n_agents <= traffic_a:
        return 0.0
    pw = erlang_c_prob_wait(n_agents, traffic_a)
    return 1 - pw * math.exp(-(n_agents - traffic_a) * target_s / aht_s)


def occupancy(n_agents: int, traffic_a: float) -> float:
    return traffic_a / n_agents if n_agents > 0 else 1.0


def agents_required(calls: float, aht_s: float, interval_s: int = 1800,
                    target_sl: float = 0.80, target_s: float = 20,
                    max_occupancy: float = 0.90) -> dict:
    """Nombre d'agents requis pour un crEneau.

    calls      : nombre d'appels prEvus sur l'intervalle
    aht_s      : temps moyen de traitement (s)
    interval_s : durEe de l'intervalle (defaut 30 min)
    Renvoie un dict avec n_agents, niveau de service atteint, occupation, charge (Erlangs).
    """
    if calls <= 0 or aht_s <= 0:
        return {"agents": 0, "trafic_erlang": 0.0, "service_level": 1.0, "occupation": 0.0}
    traffic_a = calls * aht_s / interval_s          # intensitE de trafic (Erlangs)
    n = max(1, math.ceil(traffic_a))
    while True:
        sl = service_level(n, traffic_a, aht_s, target_s)
        occ = occupancy(n, traffic_a)
        if sl >= target_sl and occ <= max_occupancy:
            break
        n += 1
        if n > 1000:
            break
    return {
        "agents": n,
        "trafic_erlang": round(traffic_a, 2),
        "service_level": round(sl, 4),
        "occupation": round(occ, 4),
        "prob_attente": round(erlang_c_prob_wait(n, traffic_a), 4),
    }


def erlang_a_adjust(n_erlang_c: int, abandon_rate: float) -> int:
    """Correction simple Erlang A : l'impatience des clients rEduit le besoin d'agents.

    Approximation pragmatique : on retire une fraction proportionnelle au taux d'abandon
    observE, bornEe pour rester prudent.
    """
    reduction = min(0.15, abandon_rate * 0.5)       # au plus 15 % de reduction
    return max(1, int(round(n_erlang_c * (1 - reduction))))


def staffing_plan(forecast_by_slot, aht_s: float, **kwargs):
    """Construit un plan d'effectifs A partir d'un dict {slot_label: volume_prevu}."""
    plan = []
    for slot, vol in forecast_by_slot.items():
        r = agents_required(vol, aht_s, **kwargs)
        plan.append({"creneau": slot, "volume_prevu": round(vol, 1),
                     "agents_requis": r["agents"], "trafic_erlang": r["trafic_erlang"],
                     "service_level": r["service_level"], "occupation": r["occupation"]})
    return plan
