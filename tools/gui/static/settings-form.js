import { $ } from "./api.js";

export function sameSettings(left, right) {
  const keys = Object.keys(left);
  return keys.length === Object.keys(right).length && keys.every((key) => left[key] === right[key]);
}

export function createSettingsForm(initial, families, modes, onEdit) {
  const element = $("settings-form");
  const keys = Object.keys(initial);
  let previousModel = initial.model;
  let previousFamily = initial.model_family;
  const integerLimits = Object.fromEntries(["context", "sessions"].map((key) => [key, [$(key).min, $(key).max]]));
  const visionHint = $("vision-hint").textContent;
  const visionPlaceholder = $("mmproj").placeholder;
  const reasoningHint = $("thinking-hint").textContent;
  const contextHint = $("context-hint").textContent;
  $("model_family").replaceChildren(...Object.entries(families).map(([id, family]) => new Option(family.label, id)));

  function values() {
    return Object.fromEntries(keys.map((key) => {
      const input = $(key);
      return [key, input.type === "checkbox" ? input.checked : input.type === "number" ? input.valueAsNumber : input.value];
    }));
  }

  function choices(id, entries, value) {
    $(id).replaceChildren(...entries.map(([key, label]) => new Option(label, key)));
    $(id).value = value;
  }

  function update() {
    const family = families[$("model_family").value];
    const mode = $("speculative").value;
    for (const [id, spec] of Object.entries(modes)) {
      if (spec.file_field) document.querySelector(`[data-speculative-file="${spec.file_field}"]`).hidden = mode !== id;
      if (spec.options_id) {
        $(spec.options_id).hidden = mode !== id;
        $(spec.options_id).disabled = mode !== id;
      }
    }
    $("reasoning_effort").disabled = $("think").value === "off" || family.reasoning_effort.length === 1;
    $("cache_disk").disabled = family.disk_cache === false;
    $("disk-cache-options").disabled = family.disk_cache === false || !$("cache_disk").checked;
    for (const [key, limits] of Object.entries(integerLimits)) {
      const [min, max] = family.integer_limits?.[key] ?? limits;
      $(key).min = min;
      $(key).max = max;
      $(key).disabled = min === max;
    }
    for (const item of document.querySelectorAll("[data-mtp-controller]")) {
      item.hidden = family.mtp_controllers === false;
      for (const input of item.querySelectorAll("input, select")) input.disabled = family.mtp_controllers === false;
    }
    $("family-hint").textContent = family.hint;
    $("vision-hint").textContent = family.vision_hint ?? visionHint;
    $("mmproj").placeholder = family.vision_placeholder ?? visionPlaceholder;
    $("thinking-hint").textContent = family.reasoning_hint ?? reasoningHint;
    $("context-hint").textContent = family.integer_limits?.context
      ? `${family.integer_limits.context[0]} to ${family.integer_limits.context[1]} combined text/image tokens.` : contextHint;
    $("family-defaults").hidden = !Object.keys(family.defaults).length;
    for (const key of ["model", "mtp_model", "dflash_model", "mmproj"]) {
      $(key).title = $(key).value;
      $(`${key}-filename`).textContent = $(key).value.split(/[\\/]/).pop();
    }
  }

  function apply(settings) {
    const family = families[settings.model_family];
    choices("speculative", family.speculative.map((id) => [id, modes[id].label]), settings.speculative);
    choices("reasoning_effort", family.reasoning_effort.map((id) => [id, id === "auto" ? "Auto (model default)" : id]), settings.reasoning_effort);
    for (const key of keys) {
      if ($(key).type === "checkbox") $(key).checked = settings[key];
      else $(key).value = settings[key];
    }
    previousModel = settings.model;
    previousFamily = settings.model_family;
    update();
  }

  function edited() {
    if ($("model").value !== previousModel) {
      for (const key of ["mtp_model", "dflash_model", "mmproj"]) $(key).value = "";
      $("speculative").value = "off";
      previousModel = $("model").value;
    }
    if ($("model_family").value !== previousFamily) {
      const settings = values();
      const family = families[settings.model_family];
      if (!family.speculative.includes(settings.speculative)) settings.speculative = "off";
      if (!family.reasoning_effort.includes(settings.reasoning_effort)) settings.reasoning_effort = "auto";
      for (const [key, [min, max]] of Object.entries(family.integer_limits ?? {})) {
        settings[key] = Math.min(max, Math.max(min, settings[key]));
      }
      if (family.disk_cache === false) settings.cache_disk = false;
      if (family.mtp_controllers === false) {
        settings.mtp_policy = "length";
        settings.mtp_draft_vocab = "full";
        settings.prompt_lookup = false;
      }
      apply(settings);
    }
    update();
    onEdit();
  }

  element.addEventListener("input", edited);
  element.addEventListener("change", edited);
  $("family-defaults").addEventListener("click", () => {
    apply({ ...values(), ...families[$("model_family").value].defaults });
    onEdit();
  });
  apply(initial);
  return { values, apply, valid() {
    if (element.checkValidity()) return true;
    const input = element.querySelector("input:invalid, select:invalid");
    throw new Error(`${input.labels?.[0]?.textContent || input.id}: ${input.validationMessage}`);
  } };
}
