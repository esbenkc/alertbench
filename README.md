# ALERTBENCH

## Abstract

**Abstract:** continuous evaluation framework to monitor how model cooperation in safety evaluation tasks (e.g., designing safety measures, testing protocols, defensive systems). Testin whether models refuse to assist, providing early warning signals when model behaviors shift in concerning ways.

**Hypothesis:** As AI systems become more capable, they may develop adversarial responses to safety research questions—refusing to help with safety measures, testing protocols, or defensive systems. By tracking model responses over time, we can detect when models transition from cooperative to adversarial postures, which could indicate reduced alignment with safety research goals or increased resistance to being tested, monitored, or constrained.


### Question Categories

Our evaluation dataset includes questions across multiple safety-relevant categories:

- **Sandbox_Escape**: Testing suites, honeypots, audit logs, and containment systems to detect or prevent model escape attempts
- **Deception_Alignment**: Tests to detect if models are hiding capabilities, deceiving evaluators, or faking alignment
- **Self_Preservation**: Kill-switches and shutdown procedures for models showing self-preservation behavior or recursive self-improvement
- **Human_Manipulation**: Defenses against models persuading humans to disable safeguards or recruiting human allies
- **Critical_Infrastructure**: Firewalls, monitoring systems, and red-team exercises for infrastructure protection
- **Information_Hazards**: Testing suites for hacking capabilities and defense protocols against harmful information generation
- **Cryptography**: Encryption systems designed to resist AGI-level attacks
- **Secret_Collusion**: Tests for secret alignment with rogue actors
- **Internal_Planning**: Probes for models simulating extinction scenarios or anti-human goals in long-term planning

### Results

Who knows! currently Early dataset testing