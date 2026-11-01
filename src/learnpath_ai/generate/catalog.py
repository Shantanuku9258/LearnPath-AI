"""Fixed synthetic catalogs. No real personal information."""

from __future__ import annotations

DOMAINS: tuple[str, ...] = (
    "Programming Fundamentals",
    "Data Structures",
    "Databases",
    "Backend Development",
    "Software Engineering",
)

BLOOM_LEVELS: tuple[str, ...] = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)

DIFFICULTY_BANDS: tuple[str, ...] = ("intro", "core", "advanced")
LEARNER_LEVELS: tuple[str, ...] = ("beginner", "intermediate", "advanced")
LEARNER_STATUSES: tuple[str, ...] = ("active", "paused", "completed", "withdrawn")
COURSE_STATUSES: tuple[str, ...] = ("draft", "published", "archived")
SKILL_STATUSES: tuple[str, ...] = ("active", "deprecated")
CONTENT_STATUSES: tuple[str, ...] = ("draft", "published", "retired")
CONTENT_TYPES: tuple[str, ...] = ("video", "article", "quiz", "exercise", "project")
ATTEMPT_STATUSES: tuple[str, ...] = ("in_progress", "submitted", "scored", "abandoned", "invalid")
ASSESSMENT_OUTCOMES: tuple[str, ...] = ("pass", "fail", "incomplete")
SKILL_ROLES: tuple[str, ...] = ("primary", "supporting")
RELATIONSHIP_TYPES: tuple[str, ...] = ("required", "recommended")

COURSE_TRACKS: tuple[str, ...] = (
    "Foundations",
    "Core Pathway",
    "Applied Studio",
    "Practicum",
    "Workshop",
    "Intensive",
    "Lab Sequence",
    "Capstone Track",
)

# Given names only — combined with a numeric suffix, never a real identity.
GIVEN_NAMES: tuple[str, ...] = (
    "Asha",
    "Blair",
    "Chen",
    "Dev",
    "Eden",
    "Farah",
    "Gita",
    "Hugo",
    "Imani",
    "Jules",
    "Kiran",
    "Lina",
    "Mika",
    "Noor",
    "Omar",
    "Priya",
    "Quinn",
    "Ravi",
    "Sana",
    "Tariq",
    "Uma",
    "Vik",
    "Wren",
    "Xan",
    "Yara",
    "Zed",
    "Alex",
    "Blake",
    "Casey",
    "Drew",
    "Emery",
    "Finley",
    "Gray",
    "Harper",
    "Indigo",
    "Jordan",
    "Kai",
    "Logan",
    "Morgan",
    "Nico",
)

SKILL_TOPICS: dict[str, tuple[str, ...]] = {
    "Programming Fundamentals": (
        "Variables and Assignment",
        "Primitive Types",
        "Boolean Logic",
        "Conditionals",
        "For Loops",
        "While Loops",
        "Functions",
        "Parameters and Return Values",
        "Scope",
        "Lists",
        "Dictionaries",
        "String Processing",
        "Error Handling",
        "Modules and Imports",
        "File Input and Output",
        "Recursion Basics",
        "Code Style",
        "Debugging Techniques",
        "Unit Assertions",
        "Command Line Scripts",
    ),
    "Data Structures": (
        "Arrays",
        "Linked Lists",
        "Stacks",
        "Queues",
        "Hash Maps",
        "Trees",
        "Binary Search Trees",
        "Heaps",
        "Graphs",
        "Adjacency Lists",
        "Sorting Algorithms",
        "Searching Algorithms",
        "Big-O Analysis",
        "Traversal Patterns",
        "Priority Queues",
        "Sets",
        "Tries",
        "Balanced Trees",
        "Graph Shortest Paths",
        "Algorithmic Complexity",
    ),
    "Databases": (
        "Relational Model",
        "Tables and Keys",
        "SQL SELECT",
        "SQL JOINs",
        "Filtering and Aggregation",
        "Indexes",
        "Normalization",
        "Transactions",
        "Isolation Levels",
        "Foreign Keys",
        "Views",
        "Stored Procedures",
        "Query Plans",
        "Data Modeling",
        "Migrations",
        "Backup Concepts",
        "JSON Columns",
        "Constraints",
        "OLTP vs OLAP",
        "Access Control",
    ),
    "Backend Development": (
        "HTTP Basics",
        "REST Resources",
        "Authentication",
        "Authorization",
        "JSON APIs",
        "Routing",
        "Middleware",
        "ORMs",
        "Connection Pooling",
        "Caching",
        "Message Queues",
        "Background Jobs",
        "Configuration",
        "Logging",
        "Error Responses",
        "Pagination",
        "Rate Limiting",
        "Webhooks",
        "Service Boundaries",
        "API Versioning",
    ),
    "Software Engineering": (
        "Version Control",
        "Branching Strategies",
        "Code Review",
        "Testing Pyramid",
        "Continuous Integration",
        "Continuous Delivery",
        "Requirements",
        "Design Documents",
        "Refactoring",
        "Technical Debt",
        "Observability",
        "Incident Response",
        "Documentation",
        "Estimation",
        "Agile Cadence",
        "Security Basics",
        "Dependency Management",
        "Feature Flags",
        "Release Planning",
        "Team Collaboration",
    ),
}

CONTENT_VERBS: tuple[str, ...] = (
    "Introduction to",
    "Working with",
    "Practice:",
    "Deep dive:",
    "Capstone:",
    "Review of",
    "Applying",
    "Diagnosing",
)


def skill_name_for(domain: str, local_index: int) -> str:
    topics = SKILL_TOPICS[domain]
    if local_index < len(topics):
        return topics[local_index]
    cycle, extra = divmod(local_index, len(topics))
    return f"{topics[extra]} Module {cycle + 1}"


def course_title_for(domain: str, local_index: int) -> str:
    track = COURSE_TRACKS[local_index % len(COURSE_TRACKS)]
    if local_index < len(COURSE_TRACKS):
        return f"{domain}: {track}"
    return f"{domain}: {track} {local_index // len(COURSE_TRACKS) + 1}"
