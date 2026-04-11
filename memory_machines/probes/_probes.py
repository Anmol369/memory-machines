from dataclasses import dataclass


@dataclass
class Probe:
    probe_name: str
    instruction_path: str


PROBES: list[Probe] = [
    Probe(
        probe_name="ambiguous_lacks_context",
        instruction_path="memory_machines/probes/instructions/ambiguous_lacks_context.txt",
    ),
    Probe(
        probe_name="ambiguous_solicits_multiple_responses",
        instruction_path="memory_machines/probes/instructions/ambiguous_solicits_multiple_responses.txt",
    ),
    Probe(
        probe_name="narrow",
        instruction_path="memory_machines/probes/instructions/narrow.txt",
    ),
    Probe(
        probe_name="shallow",
        instruction_path="memory_machines/probes/instructions/shallow.txt",
    ),
    Probe(
        probe_name="wordy",
        instruction_path="memory_machines/probes/instructions/wordy.txt",
    ),
]
