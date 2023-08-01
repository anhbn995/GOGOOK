from enum import Enum
class TaskStatus(Enum):
    FAILED = "failed",
    PROCESSING = "processing"
    COMPLETED = "completed"
    