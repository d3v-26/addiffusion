"""Baseline image generators for Phase 2 comparison (research.md §4.1)."""

from src.baselines.base import BaseBaseline, BaselineResult, load_sd15_pipeline, save_results
from src.baselines.fixed_step import (
    ALL_FIXED_STEP_BASELINES,
    DDIM20,
    DDIM50,
    DPMSolver20,
    DPMSolver50,
    Euler20,
    PNDM20,
    UniPC20,
)
from src.baselines.oracle import OracleStopBaseline
from src.baselines.random_search import RandomSearchBaseline
from src.baselines.sag import SAGBaseline
from src.baselines.attend_excite import AttendExciteBaseline
from src.baselines.lcm import LCMBaseline
from src.baselines.sdxl_turbo import SDXLTurboBaseline

__all__ = [
    "BaseBaseline",
    "BaselineResult",
    "load_sd15_pipeline",
    "save_results",
    "DDIM20",
    "DDIM50",
    "DPMSolver20",
    "DPMSolver50",
    "Euler20",
    "UniPC20",
    "PNDM20",
    "ALL_FIXED_STEP_BASELINES",
    "OracleStopBaseline",
    "RandomSearchBaseline",
    "SAGBaseline",
    "AttendExciteBaseline",
    "LCMBaseline",
    "SDXLTurboBaseline",
]
