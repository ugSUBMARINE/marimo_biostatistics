"""Check every quiz option has reachable, distinct, correctly routed feedback."""

import ast
import unittest
from pathlib import Path
from unittest.mock import patch

import marimo as mo

from companion_style import review_feedback

ROOT = Path(__file__).resolve().parents[1]


def question_specs(path):
    tree = ast.parse(path.read_text())
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    questions = {}
    for name, node in assignments.items():
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "radio"
            and any(
                keyword.arg == "value"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is None
                for keyword in node.keywords
            )
        ):
            questions[name] = ast.literal_eval(node.args[0])

    feedback = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "review_feedback"
        ):
            continue
        name = node.args[0].value.id
        kwargs = {}
        for keyword in node.keywords:
            value = keyword.value
            if isinstance(value, ast.Name):
                value = assignments[value.id]
            kwargs[keyword.arg] = ast.literal_eval(value)
        if name in feedback:
            raise AssertionError(f"Duplicate feedback for {name} in {path.name}")
        feedback[name] = kwargs
    return questions, feedback


class ReviewFeedbackTests(unittest.TestCase):
    def test_every_quiz_option_has_distinct_feedback(self):
        for path in sorted(ROOT.glob("0[1-6]_*.py")):
            questions, feedback = question_specs(path)
            with self.subTest(notebook=path.name):
                self.assertTrue(questions)
                self.assertEqual(set(questions), set(feedback))
            for name, options in questions.items():
                with self.subTest(notebook=path.name, question=name):
                    kwargs = feedback[name]
                    correct = kwargs["correct_value"]
                    incorrect = kwargs["incorrect_text"]
                    self.assertIn(correct, options.values())
                    self.assertEqual(len(options), len(set(options.values())))
                    self.assertIsInstance(incorrect, dict)
                    self.assertEqual(set(incorrect), set(options.values()) - {correct})
                    self.assertEqual(len(set(incorrect.values())), len(incorrect))
                    self.assertTrue(all(text.strip() for text in incorrect.values()))

                    # Exercise real radio label -> value conversion and Markdown
                    # rendering, observing the callout's message and status.
                    for label, value in [(None, None), *options.items()]:
                        widget = mo.ui.radio(options, value=label)
                        self.assertEqual(widget.value, value)
                        with patch("companion_style.mo.callout") as callout:
                            review_feedback(widget.value, **kwargs)
                        body = callout.call_args.args[0].text
                        kind = callout.call_args.kwargs["kind"]
                        if value is None:
                            expected = kwargs.get(
                                "unanswered_text", "Select an answer above."
                            )
                            self.assertEqual(kind, "neutral")
                        elif value == correct:
                            expected = kwargs["correct_text"]
                            self.assertEqual(kind, "success")
                        else:
                            expected = incorrect[value]
                            self.assertEqual(kind, "danger")
                        self.assertEqual(body, mo.md(expected).text)

    def test_existing_single_message_feedback_is_supported(self):
        with patch("companion_style.mo.callout") as callout:
            review_feedback(
                "wrong",
                correct_value="right",
                correct_text="Correct.",
                incorrect_text="Reconsider the assumptions.",
            )
        self.assertEqual(callout.call_args.kwargs["kind"], "danger")
        self.assertIn("Reconsider the assumptions.", callout.call_args.args[0].text)


if __name__ == "__main__":
    unittest.main()
