"""Timing must remain serializable without blocking on unfinished GPU work."""
import json
from unittest.mock import Mock

from act_rlt.stage2_timing import Stage2Timing


def test_pending_cuda_event_is_not_synchronized():
    timing = Stage2Timing()
    start, end = Mock(), Mock()
    end.query.return_value = False
    timing.events["act"] = (start, end)
    assert timing.resolve() == {"act_cuda_ms": None}
    start.elapsed_time.assert_not_called()
    start.synchronize.assert_not_called()
    end.synchronize.assert_not_called()
    end.query.return_value = True
    start.elapsed_time.return_value = 12.5
    assert timing.resolve() == {"act_cuda_ms": 12.5}
    end.synchronize.assert_not_called()


def test_cpu_measurement_is_json_serializable():
    timing = Stage2Timing()
    with timing.measure("act"):
        pass
    values = timing.resolve()
    assert values["act_wall_ms"] >= 0
    assert "act_cuda_ms" not in values
    json.dumps(values)
