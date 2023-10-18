import random

from easyfsl.samplers import TaskSampler


class CustomTaskSampler(TaskSampler):
    def __init__(self, dataset, n_way, n_shot, n_query, n_tasks):
        super().__init__(dataset, n_way, n_shot, n_query, n_tasks)
        self.dataset = dataset

    def __iter__(self):
        for _ in range(self.n_tasks):
            # Select n_way classes
            selected_classes = random.sample(self.dataset.classes, self.n_way)

            # Select n_shot + n_query examples per class
            selected_examples = [
                random.sample(self.dataset.get_class_examples(class_name), self.n_shot + self.n_query)
                for class_name in selected_classes
            ]

            # Split support and query sets
            support_set = [examples[: self.n_shot] for examples in selected_examples]
            query_set = [examples[self.n_shot:] for examples in selected_examples]

            # Flatten lists of lists
            support_set = [example for class_examples in support_set for example in class_examples]
            query_set = [example for class_examples in query_set for example in class_examples]

            yield support_set + query_set