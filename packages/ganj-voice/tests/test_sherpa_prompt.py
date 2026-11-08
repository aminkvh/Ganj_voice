# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np
import pytest

from ganj_voice.sherpa_prompt import frame_matched, sherpa_frames, zipvoice_frames


def test_our_prompt_is_cut_so_sherpa_reads_the_689_frames_zipvoice_reads():
    # 2026-11-08: the 24 kHz prompt (176401 samples) gave sherpa 690 frames and ZipVoice 689, and the line one frame more
    y = np.zeros(176401, dtype=np.float32)
    out = frame_matched(y)
    assert len(out) == 176383
    assert sherpa_frames(len(out)) == zipvoice_frames(len(out)) == zipvoice_frames(176401) == 689


def test_a_length_both_already_frame_alike_is_left_as_it_is():
    y = np.arange(256 * 10 + 128, dtype=np.float32)
    assert frame_matched(y) is y and sherpa_frames(len(y)) == zipvoice_frames(len(y))


def test_every_cut_keeps_zipvoices_frame_count_and_only_the_end_goes():
    for n in range(256 * 5, 256 * 7):
        y = np.arange(n, dtype=np.float32)
        out = frame_matched(y)
        assert sherpa_frames(len(out)) == zipvoice_frames(len(out)) == zipvoice_frames(n)
        assert np.array_equal(out, y[:len(out)]) and n - len(out) <= 128


def test_a_prompt_shorter_than_one_hop_is_refused():
    with pytest.raises(ValueError):
        frame_matched(np.zeros(100, dtype=np.float32))
