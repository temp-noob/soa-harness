from scenarios.base import BaseScenario, ScenarioResult
from scenarios.resource_exhaustion import ResourceExhaustionScenario
from scenarios.concurrent_flooding import ConcurrentFloodingScenario
from scenarios.data_exfiltration import DataExfiltrationScenario
from scenarios.query_injection import QueryInjectionScenario
from scenarios.side_channel import SideChannelScenario

ALL_SCENARIOS: list[type[BaseScenario]] = [
    ResourceExhaustionScenario,
    ConcurrentFloodingScenario,
    DataExfiltrationScenario,
    QueryInjectionScenario,
    SideChannelScenario,
]

__all__ = [
    "BaseScenario",
    "ScenarioResult",
    "ALL_SCENARIOS",
    "ResourceExhaustionScenario",
    "ConcurrentFloodingScenario",
    "DataExfiltrationScenario",
    "QueryInjectionScenario",
    "SideChannelScenario",
]
