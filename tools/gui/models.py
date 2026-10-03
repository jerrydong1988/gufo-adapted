"""Curated UI families, not a claim of compatibility with a selected artifact."""

REASONING_LEVELS = ("auto", "low", "medium", "xhigh")
SPECULATIVE_MODES = {
    "off": {"label": "Off", "file_field": None, "options_id": None},
    "mtp": {"label": "MTP", "file_field": "mtp_model", "options_id": "mtp-options"},
    "dflash2": {"label": "DFlash2", "file_field": "dflash_model", "options_id": None},
}
MODEL_FAMILIES = {
    "generic": {
        "label": "Other / existing configuration",
        "speculative": ("off", "mtp", "dflash2"),
        "reasoning_effort": REASONING_LEVELS,
        "defaults": {},
        "hint": "Choose options supported by your model. Gufo checks artifact compatibility when loading.",
    },
    "qwen38_flash_next": {
        "label": "Qwen3.8-Flash-Next",
        "speculative": ("off", "mtp"),
        "reasoning_effort": REASONING_LEVELS,
        "defaults": {"think": "on", "reasoning_effort": "auto", "temperature": 1.0,
                     "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        "hint": "MTP needs a matching shared Q8_0 sidecar. Gufo checks projector compatibility when loading.",
    },
    "qwen38_27b": {
        "label": "Qwen3.8-27B",
        "speculative": ("off", "dflash2"),
        "reasoning_effort": REASONING_LEVELS,
        "defaults": {"think": "auto", "reasoning_effort": "auto"},
        "hint": "DFlash2 needs a matching draft and uses Gufo's adaptive defaults. Sampling remains editable.",
    },
    "gemma4": {
        "label": "Gemma 4 31B",
        "speculative": ("off", "mtp"),
        "reasoning_effort": ("auto",),
        "integer_limits": {"context": (2, 4096), "sessions": (1, 1)},
        "disk_cache": False,
        "mtp_controllers": False,
        "defaults": {"context": 4096, "sessions": 1, "think": "off", "reasoning_effort": "auto",
                     "speculative": "off", "cache_disk": False, "draft_tokens": 2,
                     "mtp_policy": "length", "mtp_draft_vocab": "full", "prompt_lookup": False},
        "hint": "Supports the qualified conventional and QAT 31B sets. Up to 4K text/image tokens, one session; "
                "disk cache is unavailable. MTP is optional and currently slower. Sampling remains editable.",
        "vision_hint": "Images require the matching BF16 projector from the main model's folder. Leave blank for text only.",
        "vision_placeholder": "Matching BF16 projector",
        "reasoning_hint": "Use Thinking On or Off. Gemma has no adjustable thinking levels.",
    },
}


def validate_model_settings(settings):
    family = MODEL_FAMILIES[settings["model_family"]]
    if settings["speculative"] not in family["speculative"]:
        raise ValueError(f"{family['label']} does not support {settings['speculative']} speculative decoding.")
    if settings["reasoning_effort"] not in family["reasoning_effort"]:
        raise ValueError(f"Unsupported reasoning level for {family['label']}.")
    for key, (low, high) in family.get("integer_limits", {}).items():
        if not low <= settings[key] <= high:
            raise ValueError(f"{family['label']}: {key} must be from {low} to {high}.")
    if family.get("disk_cache") is False and settings["cache_disk"]:
        raise ValueError(f"{family['label']} does not support disk continuation caching.")
    if family.get("mtp_controllers") is False and settings["speculative"] == "mtp":
        if settings["mtp_policy"] != "length" or settings["mtp_draft_vocab"] != "full" or settings["prompt_lookup"]:
            raise ValueError(f"{family['label']} does not support Flash-Next MTP controllers or prompt lookup.")
