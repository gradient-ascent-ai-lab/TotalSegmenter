from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from totalsegmenter.enums import MaskType, NnUnetModelType
from totalsegmenter.settings import settings


class Task(BaseModel):
    name: str
    task_id: int | None = None
    notes: str = ""

    resample: float | None = None
    mask_type: MaskType | None = None
    crop_addon: list[int] = [3, 3, 3]

    model_type: NnUnetModelType
    trainer: str | None = "nnUNetTrainerV2"
    weights_url: str | None = None
    weights_subdir: str | None = None
    folds: list[int] | None = None

    class_map: dict[int, str] | None = None
    class_map_v2: dict[int, str] | None = None
    preview_roi_groups: list[list[str]] | None = None

    subtasks: list["Task"] | None = None

    def get_class_map(self) -> dict[int, str]:
        return self.class_map or self.class_map_v2

    def get_model_weights_dir(self) -> Path:
        if self.weights_subdir is None:
            raise ValueError(f"Task {self.name} has no weights_subdir")

        model_parent_dir = settings.WEIGHTS_DIR / self.weights_subdir
        potential_model_weights_dir = next(model_parent_dir.glob("**/plans.pkl"), None)
        if not potential_model_weights_dir:
            raise ValueError(f"Could not find model weights dir in {model_parent_dir} for task {self.name}")
        return potential_model_weights_dir.parent

    def is_available(self) -> bool:
        if self.subtasks:
            return all(subtask.is_available() for subtask in self.subtasks)

        if self.weights_url is not None:
            return True

        if not (settings.WEIGHTS_DIR / self.weights_subdir).exists():
            return False

        return True


class TaskManager:
    def __init__(self):
        self.tasks_by_name: dict[str, Task] = {}
        self.tasks_by_id: dict[int, Task] = {}

        for task_path in settings.TASKS_DIR.glob("**/*.yaml"):
            with open(task_path, "r") as task_file:
                task = Task(**yaml.safe_load(task_file))
                if task.task_id:
                    self.tasks_by_id[task.task_id] = task
                if task.subtasks:
                    for subtask in task.subtasks:
                        self.tasks_by_name[subtask.name] = subtask
                        self.tasks_by_id[subtask.task_id] = subtask
                self.tasks_by_name[task.name] = task

    def get_task_by_name(self, task_name: str) -> Task:
        task = self.tasks_by_name.get(task_name, None)
        if task is None:
            raise ValueError(f"Task {task_name} not found")

        if not task.is_available():
            raise ValueError(f"Task {task_name} is not available")

        return task

    def get_task_by_id(self, task_id: int) -> Task:
        task = self.tasks_by_id.get(task_id, None)
        if task is None:
            raise ValueError(f"Task {task_id} not found")

        if not task.is_available():
            raise ValueError(f"Task {task_id} is not available")

        return task

    def list_tasks(self) -> list[Task]:
        return list(self.tasks_by_name.values())


task_manager = TaskManager()
