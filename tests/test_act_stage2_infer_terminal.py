from io import StringIO
from types import SimpleNamespace

import pytest
import torch

from act_rlt.infer_stage2 import run_episodes
from act_rlt.infer_stage2_terminal import InferenceTerminal
from act_rlt.train_stage2 import FrankaInsertionStage2Env


def result(*, steps=4, clipped=0, boundary=10.0, done=False, terminated=False,
           truncated=False, stopped=False, **info):
    return SimpleNamespace(
        actual_steps=steps, next_batch={}, done=done, terminated=terminated,
        truncated=truncated, stop_requested=stopped,
        info={"safety_clip_steps": clipped, "boundary_inference_ms": boundary,
              "episode_elapsed_s": 1.0, **info},
    )


def test_inference_output_aggregates_clipping_and_does_not_report_pending_failure(monkeypatch):
    stream = StringIO()
    terminal = InferenceTerminal(1, 5, stream=stream)
    chunks = iter([
        result(clipped=1, boundary=9),
        result(boundary=11),
        result(clipped=2, boundary=13, done=True, terminated=True, success=True),
    ])
    calls = []

    def execute(*args, **kwargs):
        assert kwargs["deterministic"] is True
        assert kwargs["warmup"] is False
        calls.append(kwargs["remaining_steps"])
        return next(chunks)

    env = SimpleNamespace(
        episode_time_s=5, period_s=1 / 30, inference_terminal=terminal,
        reset=lambda **kwargs: {}, execute_policy_chunk=execute,
    )
    monkeypatch.setattr("act_rlt.infer_stage2_terminal.time.monotonic", lambda: 0.0)
    run_episodes(env, SimpleNamespace(config=SimpleNamespace(chunk_length=4)), 1)
    text = stream.getvalue()
    assert text.count("  RUN ") == 1
    assert "success=False" not in text
    assert "[SUCCESS] Episode 01/01" in text
    assert "12 steps | clipped 3/12 | boundary avg 11.0 ms" in text
    assert "success rate 100.0%" in text
    assert calls == [1000, 996, 992]


def test_inference_summary_distinguishes_failure_workspace_and_stop():
    stream = StringIO()
    terminal = InferenceTerminal(5, 5, stream=stream)
    results = iter([
        result(done=True, terminated=True, success=True),
        result(done=True, terminated=True, success=False, timed_out=True),
        result(steps=0, done=True, truncated=True, workspace_violation=True,
               workspace_error="target Z outside [0.35, 0.45]"),
        result(done=True, truncated=True, stopped=True),
    ])
    resets = []
    env = SimpleNamespace(
        episode_time_s=5, period_s=1 / 30, inference_terminal=terminal,
        reset=lambda **kwargs: resets.append(kwargs["episode_id"]) or {},
        execute_policy_chunk=lambda *args, **kwargs: next(results),
    )
    run_episodes(env, SimpleNamespace(config=SimpleNamespace(chunk_length=4)), 5)
    text = stream.getvalue()
    assert "[FAILURE]" in text and "time limit reached" in text
    assert "[WORKSPACE]" in text and "Reason: target Z outside" in text
    assert "[STOPPED]" in text
    assert "Session stopped | 3 completed episodes | success rate 33.3%" in text
    assert "success 1 | failure 1 | workspace 1 | stopped 1" in text
    assert resets == [0, 1, 2, 3]


def test_terminal_throttles_progress_and_clears_it_before_prompts(monkeypatch):
    class TTY(StringIO):
        def isatty(self):
            return True

    stream = TTY()
    terminal = InferenceTerminal(2, 5, stream=stream)
    clock = [0.0]
    monkeypatch.setattr("act_rlt.infer_stage2_terminal.time.monotonic", lambda: clock[0])
    terminal.status(0, 0.1, 4, 0, 9)
    clock[0] = 0.5
    terminal.status(0, 0.5, 16, 1, 9)
    clock[0] = 1.1
    terminal.status(0, 1.1, 32, 2, 9)
    prompt = terminal.prompt("Enter: start | q: exit")
    assert stream.getvalue().count("  RUN ") == 2
    assert terminal._line_width == 0
    assert stream.getvalue().endswith("\r")
    assert prompt == "  Enter: start | q: exit\n> "


def test_interrupted_reset_keeps_keyboard_interrupt_and_summarizes_session():
    stream = StringIO()
    terminal = InferenceTerminal(1, 5, stream=stream)

    def reset(**kwargs):
        raise KeyboardInterrupt

    env = SimpleNamespace(episode_time_s=5, period_s=1 / 30,
                          inference_terminal=terminal, reset=reset)
    with pytest.raises(KeyboardInterrupt):
        run_episodes(env, SimpleNamespace(config=SimpleNamespace(chunk_length=4)), 1)
    assert "Session interrupted | 0 completed episodes | success rate n/a" in stream.getvalue()


@pytest.mark.parametrize("key", ["s", "f", "q"])
def test_inference_timeout_prompt_preserves_labels_and_retreat_behavior(monkeypatch, key):
    stream = StringIO()
    terminal = InferenceTerminal(1, 5, stream=stream)
    execution = result(done=True, timed_out=True, collector_paused=True, workspace_violation=False)
    execution.reward_seq = torch.zeros(4)
    execution.intervention = False
    events = []
    env = object.__new__(FrankaInsertionStage2Env)
    env.inference_terminal = terminal
    env._episode_start = 0.0
    env._monitor = None
    env._runtime = SimpleNamespace(
        warmup=False, deterministic=True, publish_actor=lambda actor: 0.0,
        next_result=lambda: execution, close=lambda: events.append("close"),
    )
    env._retreat_after_outcome = lambda: events.append("retreat")
    prompts = []
    choices = iter(["invalid", key])

    def choose(prompt):
        prompts.append(prompt)
        return next(choices)

    monkeypatch.setattr("builtins.input", choose)
    returned = env.execute_policy_chunk(
        SimpleNamespace(actor=object()), {}, warmup=False, remaining_steps=1000, deterministic=True,
    )
    assert returned is execution
    assert execution.reward_seq[-1].item() == float(key == "s")
    assert bool(execution.info.get("success")) == (key == "s")
    assert execution.terminated == (key != "q")
    assert execution.stop_requested == (key == "q")
    assert events == ["close"] + (["retreat"] if key != "q" else [])
    assert len(prompts) == 2 and all("training" not in p for p in prompts)
    assert "TIME LIMIT | motion stopped" in stream.getvalue()
    assert "Choose s, f, or q." in stream.getvalue()
