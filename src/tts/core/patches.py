"""Runtime patches for Hugging Face Hub and SpeechBrain."""
from __future__ import annotations

import warnings


def configure_runtime_warning_filters() -> None:
    """Register filters before SpeechBrain / torchaudio are imported."""
    warnings.filterwarnings(
        "ignore",
        message=r".*torchaudio\._backend\.list_audio_backends has been deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"speechbrain\.utils\.torch_audio_backend",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*speechbrain\.pretrained.*was deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.cuda\.amp\.custom_fwd.*is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*You are sending unauthenticated requests to the HF Hub.*",
        category=UserWarning,
    )


def patch_torchaudio_list_backends() -> None:
    """Avoid deprecated ``list_audio_backends`` when SpeechBrain probes backends."""
    try:
        import torchaudio

        def _list_audio_backends():
            try:
                import soundfile  # noqa: F401

                return ["soundfile"]
            except ImportError:
                return []

        torchaudio.list_audio_backends = _list_audio_backends
    except Exception:
        pass


def patch_hf_hub_use_auth_token_compat() -> None:
    try:
        import huggingface_hub

        _orig = huggingface_hub.hf_hub_download

        def _hf_hub_download_compat(*args, **kwargs):
            if "use_auth_token" in kwargs and kwargs.get("token") is None:
                kwargs["token"] = kwargs.pop("use_auth_token")
            else:
                kwargs.pop("use_auth_token", None)
            return _orig(*args, **kwargs)

        huggingface_hub.hf_hub_download = _hf_hub_download_compat
        try:
            import huggingface_hub.file_download as _fd

            if hasattr(_fd, "hf_hub_download"):
                _fd.hf_hub_download = _hf_hub_download_compat
        except Exception:
            pass
    except Exception:
        pass


def patch_speechbrain_fetch_optional_custom_py() -> None:
    try:
        import speechbrain.utils.fetching as _sb_fetch

        _orig = _sb_fetch.fetch

        def _fetch(*args, **kwargs):
            try:
                return _orig(*args, **kwargs)
            except Exception as exc:
                fn = kwargs.get("filename")
                if fn is None and args:
                    fn = args[0]
                if str(fn) != "custom.py":
                    raise
                msg = str(exc).lower()
                tnm = type(exc).__name__.lower()
                if (
                    "404" in msg
                    or "not found" in msg
                    or "remoteentry" in tnm
                    or "entrynotfound" in tnm
                    or "httpstatus" in tnm
                    or "http error" in msg
                ):
                    raise ValueError("optional custom.py not found on Hugging Face Hub") from exc
                raise

        _sb_fetch.fetch = _fetch
        for mod_name in (
            "speechbrain.inference.interfaces",
            "speechbrain.inference.classifiers",
            "speechbrain.utils.parameter_transfer",
        ):
            try:
                mod = __import__(mod_name, fromlist=["fetch"])
                mod.fetch = _fetch
            except Exception:
                pass
    except Exception:
        pass


def apply_runtime_patches() -> None:
    patch_torchaudio_list_backends()
    patch_hf_hub_use_auth_token_compat()
    patch_speechbrain_fetch_optional_custom_py()
