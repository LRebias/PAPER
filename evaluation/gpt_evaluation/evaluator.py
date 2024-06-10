import os

class Evaluator:
    def __init__(self, type) -> None:
        assert type in ['consistency', 'fluency', 'persona_consistency','interestingness'], "error type! please check it!"
        self.type = type
        template_path = os.path.join('gpt_evaluation','prompt', f"{type}.txt")
        self.template = open(template_path).read()

    def make_queries(self, inputs, predicts):
        def make_query(input, predict):
            conv = input

            template = self.template.replace(
                '{{conversation history}}', conv
            ).replace(
                '{{response}}', predict
            )

            return template

        queries = []
        for input,predict in zip(inputs, predicts):
            query = make_query(
                input=input,
                predict=predict
            )

            queries.append(query)

        return queries

