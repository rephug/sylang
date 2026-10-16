"""The probes must pass on the reference model and fail on each injected defect.

Run from this directory:  python -m unittest test_probes -v
"""

from __future__ import annotations

import io
import random
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for the repository
    torch = None

if torch is not None:
    import probes
    import tasks
    from model import PAD, VOCAB_SIZE, ModelConfig, TinyByteLM, answer_loss, greedy_generate


def _model(bug=None, seed=0):
    torch.manual_seed(seed)
    return TinyByteLM(ModelConfig(d_model=64, n_layers=2, n_heads=4, bug=bug)).eval()


def _batch(bug=None):
    rng = random.Random(0)
    examples = [tasks.role_select_example(rng, tasks.FreshIds("train"), select=True) for _ in range(3)]
    examples.append(tasks.copy_none_example(rng, tasks.FreshIds("train"), copy=True))  # shorter row
    return tasks.collate(examples, bug=bug)


def _loss(logits, batch):
    return answer_loss(logits, batch["labels"])


@unittest.skipIf(torch is None, "torch not installed")
class MechanicalProbes(unittest.TestCase):
    def setUp(self):
        self.ids = _batch()["input_ids"][:1, :40]

    def test_causality(self):
        self.assertTrue(probes.causality_probe(_model(), self.ids, VOCAB_SIZE).passed)
        self.assertFalse(probes.causality_probe(_model("noncausal"), self.ids, VOCAB_SIZE).passed)

    def test_visibility(self):
        query = self.ids.shape[1] - 1
        self.assertTrue(probes.visibility_probe(_model(), self.ids, query, VOCAB_SIZE).passed)
        self.assertFalse(probes.visibility_probe(_model("window2"), self.ids, query, VOCAB_SIZE).passed)
        # A stop-gradient defect is invisible to forward probes...
        self.assertTrue(probes.visibility_probe(_model("detach_kv"), self.ids, query, VOCAB_SIZE).passed)

    def test_gradient_reach(self):
        batch = _batch()
        for bug, expected in ((None, True), ("window2", False), ("detach_kv", False)):
            model = _model(bug).train()
            result = probes.gradient_reach_probe(model, model.tok_emb, batch, _loss)
            self.assertEqual(result.passed, expected, (bug, result))
        # Left pads are excluded when pad_id is given.
        model = _model().train()
        left = _batch("left_pad")
        self.assertTrue(probes.gradient_reach_probe(model, model.tok_emb, left, _loss, pad_id=PAD).passed)

    def test_parameter_gradients(self):
        model = _model().train()
        batch = _batch()
        optimizer = torch.optim.AdamW(model.parameters())
        loss = _loss(model(batch["input_ids"]), batch)
        self.assertTrue(probes.parameter_gradient_probe(model, loss, optimizer).passed)
        partial = torch.optim.AdamW([p for n, p in model.named_parameters() if not n.startswith("pos_emb")])
        loss = _loss(model(batch["input_ids"]), batch)
        result = probes.parameter_gradient_probe(model, loss, partial)
        self.assertFalse(result.passed)
        self.assertEqual(result.detail["not_in_optimizer"], ["pos_emb.weight"])

    def test_oracle_labels(self):
        self.assertTrue(probes.oracle_label_probe(_loss, _batch(), VOCAB_SIZE, PAD).passed)
        self.assertTrue(probes.oracle_label_probe(_loss, _batch("left_pad"), VOCAB_SIZE, PAD).passed)
        self.assertFalse(probes.oracle_label_probe(_loss, _batch("unshifted"), VOCAB_SIZE, PAD).passed)

    def test_oracle_labels_inputs_without_final_token(self):
        # Convention x = seq[:-1], labels[t] = seq[t+1] (used by many trainers).
        full = [tasks.encode(ex) for ex in (tasks.role_select_example(random.Random(i), tasks.FreshIds("train"))
                                           for i in range(3))]
        width = max(len(ids) for ids, _ in full) - 1
        x = torch.full((3, width), PAD)
        y = torch.full((3, width), -100)
        for row, (ids, start) in enumerate(full):
            x[row, : len(ids) - 1] = torch.tensor(ids[:-1])
            for t in range(start - 1, len(ids) - 1):
                y[row, t] = ids[t + 1]
        batch = {"input_ids": x, "labels": y}
        sequences = [ids for ids, _ in full]
        self.assertTrue(probes.oracle_label_probe(_loss, batch, VOCAB_SIZE, PAD, sequences=sequences).passed)

    def test_checkpoint_roundtrip(self):
        model = _model(seed=1)

        def save(m):
            buffer = io.BytesIO()
            torch.save(m.state_dict(), buffer)
            buffer.seek(0)
            return buffer

        def load(m, buffer):
            m.load_state_dict(torch.load(buffer))

        def load_renamed(m, buffer):
            # A renamed key plus strict=False silently keeps fresh weights.
            state = torch.load(buffer)
            state["position.weight"] = state.pop("pos_emb.weight")
            m.load_state_dict(state, strict=False)

        build = lambda: _model(seed=2)  # noqa: E731
        self.assertTrue(probes.checkpoint_roundtrip_probe(model, build, save, load, self.ids).passed)
        self.assertFalse(probes.checkpoint_roundtrip_probe(model, build, save, load_renamed, self.ids).passed)

    def test_padding_invariance(self):
        rng = random.Random(0)
        short = tasks.copy_none_example(rng, tasks.FreshIds("train"), copy=True)
        longer = [tasks.role_select_example(rng, tasks.FreshIds("train"), select=True) for _ in range(3)]
        model = _model()

        def logits(batch):
            return model(batch["input_ids"])

        right = probes.padding_invariance_probe(tasks.collate, logits, [short, *longer])
        left = probes.padding_invariance_probe(lambda ex: tasks.collate(ex, bug="left_pad"), logits, [short, *longer])
        self.assertTrue(right.passed, right)
        self.assertFalse(left.passed, left)

    def test_decoder_consistency(self):
        model = _model(seed=3)
        with torch.no_grad():  # sharpen the untrained model so argmax is context sensitive
            model.head.weight.mul_(50)
        prompts = [tasks.role_select_example(random.Random(i), tasks.FreshIds("test")).prompt for i in range(6)]

        def generate(prompt):
            return greedy_generate(model, [tasks.prompt_ids(prompt)], max_new=6)[0]

        def generate_without_bos(prompt):
            return greedy_generate(model, [list(prompt.encode())], max_new=6)[0]

        self.assertTrue(probes.decoder_consistency_probe(generate, model, tasks.prompt_ids, prompts).passed)
        self.assertFalse(probes.decoder_consistency_probe(generate_without_bos, model, tasks.prompt_ids, prompts).passed)


@unittest.skipIf(torch is None, "torch not installed")
class TaskInvariants(unittest.TestCase):
    def test_copy_none_lengths_match(self):
        rng = random.Random(0)
        ids = tasks.FreshIds("train")
        lengths = {len(tasks.copy_none_example(rng, ids, copy=c).prompt) for c in (True, False) for _ in range(50)}
        self.assertEqual(len(lengths), 1)

    def test_role_select_lengths_match(self):
        rng = random.Random(0)
        ids = tasks.FreshIds("train")
        lengths = {len(tasks.role_select_example(rng, ids, select=s).prompt) for s in (True, False) for _ in range(50)}
        self.assertEqual(len(lengths), 1)

    def test_heldout_ids_are_unseen_combinations_of_seen_characters(self):
        train, test = set(tasks.FreshIds("train").ids), set(tasks.FreshIds("test").ids)
        self.assertFalse(train & test)
        for position in range(3):
            self.assertEqual({i[position] for i in test} - {i[position] for i in train}, set())

    def test_fresh_endings_not_predictable_from_first_char(self):
        ids = tasks.FreshIds("train").ids
        endings = {i[1:] for i in ids if i[0] == "a"}
        self.assertGreater(len(endings), 150)

    def test_labels_are_answer_bytes_then_eos(self):
        rng = random.Random(0)
        example = tasks.role_select_example(rng, tasks.FreshIds("train"), select=True)
        batch = tasks.collate([example])
        labels = batch["labels"][0]
        supervised = labels[labels != -100].tolist()
        self.assertEqual(supervised[:-1], list(example.answer.encode()))
        ids = batch["input_ids"][0]
        positions = (labels != -100).nonzero().flatten()
        self.assertTrue(torch.equal(labels[positions], ids[positions + 1]))

    def test_swap_and_replace(self):
        rng = random.Random(0)
        example = tasks.role_select_example(rng, tasks.FreshIds("train"), select=True)
        swapped = tasks.swap_roles(example)
        self.assertEqual(sorted(swapped.prompt), sorted(example.prompt))
        self.assertEqual(swapped.targets, example.targets[::-1])
        renamed = tasks.replace_identifiers(example, {example.targets[0]: "q0q"})
        self.assertIn("q0q", renamed.prompt)
        self.assertTrue(renamed.answer.startswith("q0q/"))


if __name__ == "__main__":
    unittest.main()
