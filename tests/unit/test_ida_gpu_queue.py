"""Verify the shared IDA scheduler can keep four model workers active."""

from pathlib import Path
import threading
import time

from scripts.ida_gpu_queue import execute_queue


def test_four_gpus_are_used_across_studies(tmp_path, monkeypatch):
    """The fourth GPU gets a second study rather than waiting for three models."""
    for name in ('chronos', 't0', 'timesfm'):
        python = tmp_path / name / 'bin/python'
        python.parent.mkdir(parents=True)
        python.touch()
    state = {'active': 0, 'peak': 0, 'jobs': 0}
    mutex = threading.Lock()
    barrier = threading.Barrier(4)
    class Child:
        def __init__(self, command, **kwargs):
            self.first = False
            with mutex:
                state['active'] += 1
                state['peak'] = max(state['peak'], state['active'])
                state['jobs'] += 1
                self.first = state['jobs'] <= 4
        def wait(self):
            if self.first:
                barrier.wait(timeout=3)
            time.sleep(0.005)
            with mutex:
                state['active'] -= 1
            return 0
        def poll(self):
            return 0
    monkeypatch.setattr('scripts.ida_gpu_queue.subprocess.Popen', Child)
    monkeypatch.setattr('scripts.ida_gpu_queue.signal.signal', lambda *args: None)
    studies = [tmp_path / f'study-{index}' for index in range(3)]
    execute_queue(studies, ['chronos2', 't0_gamma', 'timesfm3'],
                  ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], tmp_path, 8, 1)
    assert state['jobs'] == 9
    assert state['peak'] == 4
    assert state['active'] == 0
