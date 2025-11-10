# Ultralytics HUB stubs for offline/local installations.
#
# The upstream Ultralytics package ships a full HUB client that enables
# cloud-connected training. In this trimmed repo the module is absent, but
# core training still imports a few symbols unconditionally. Providing this
# lightweight shim keeps local training working without the HUB extras.

HUB_WEB_ROOT = "https://hub.ultralytics.com"
PREFIX = "[HUB] "


class HUBTrainingSession:  # pragma: no cover
    """Minimal placeholder that raises if HUB-specific features are requested."""

    @classmethod
    def create_session(cls, *args, **kwargs):
        raise ImportError(
            "Ultralytics HUB is not available in this environment. "
            "Install the full `ultralytics` package with HUB support if you "
            "need to sync runs to hub.ultralytics.com."
        )
