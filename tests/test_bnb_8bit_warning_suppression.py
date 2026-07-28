"""Regression test for the MatMul8bitLt cast-warning flood on
preset_8bit_midpoint's log. bitsandbytes emits this on EVERY int8 matmul
call, and different bitsandbytes versions route it through either
warnings.warn() (confirmed locally, 0.49.2) or logging.getLogger(
"bitsandbytes.autograd._functions").warning() (the format actually seen
on Colab, WARNING:bitsandbytes.autograd._functions:...).

Only the logging.Filter half is asserted here: pytest's own warnings
plugin resets/reorders warnings.filters around every test (confirmed --
a warnings.filterwarnings()-based assertion failed under pytest even
though the same check passed in a plain `python -c` script), so a
warnings-based fix is NOT reliably testable, and turned out not to be
reliably WORKING either -- this is very likely why the message kept
recurring after the first fix (which only added a warnings.filterwarnings
call): something (pytest here, possibly a Colab library/notebook
extension there) wiped it before bitsandbytes ever got a chance to warn.
The logging.Filter, registered on the logger object itself rather than
the global mutable warnings.filters list, is not subject to that reset
and is the fix this now actually depends on.
"""
import logging

import steering_factory.model_utils  # noqa: F401 -- import registers the logging filter


def test_logging_path_is_suppressed(caplog):
    bnb_logger = logging.getLogger("bitsandbytes.autograd._functions")
    with caplog.at_level(logging.WARNING, logger="bitsandbytes.autograd._functions"):
        bnb_logger.warning("MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization")
    assert not any("MatMul8bitLt: inputs will be cast from" in r.message for r in caplog.records)


def test_logging_path_does_not_suppress_unrelated_messages(caplog):
    bnb_logger = logging.getLogger("bitsandbytes.autograd._functions")
    with caplog.at_level(logging.WARNING, logger="bitsandbytes.autograd._functions"):
        bnb_logger.warning("some other real bnb warning")
    assert any("some other real bnb warning" in r.message for r in caplog.records)
