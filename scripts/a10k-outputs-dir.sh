#!/usr/bin/env bash

# Reproducibility mirror for A10K (10k-astral-dataset) run outputs.
#
# run-a10k.sh keeps writing every result inside the dataset tree exactly as
# before:
#
#   <data>/10k-simphy/<replicate>/<method>_outputs/<tree_type>/<setting>/...
#
# The functions below additionally maintain a compact, shareable copy that
# holds only the small artifacts (inferred trees, CSVs, command records, run
# logs) and never the gene trees or true species trees:
#
#   <outputs>/<method>_outputs/10k-simphy/<replicate>/<tree_type>/<setting>/...
#   <outputs>/<method>_outputs/10k-simphy/<replicate>/estimatedgenetrees.rooted.command
#   <outputs>/<dataset>.command | .source | .params | README*   (when present)
#   <outputs>/a10k_astralx_scores_merged.csv                    (collector output)
#
# <outputs> defaults to "<parent>/outputs/<dataset>", i.e. an "outputs"
# directory next to the dataset that holds a sub-directory named after it:
# "$PHYLOGENY_DATA_DIR/10k-astral-dataset" mirrors into
# "$PHYLOGENY_DATA_DIR/outputs/10k-astral-dataset", exactly where the SimPhy
# mirror lives ("$PHYLOGENY_DATA_DIR/outputs/simphy"). An explicit outputs
# directory always wins.

A10K_OUTPUTS_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Shared primitives: error printing, containment check, atomic copies.
# shellcheck source=simphy-outputs-dir.sh
source "${A10K_OUTPUTS_HELPER_DIR}/simphy-outputs-dir.sh"

# Name of the replicate container inside the dataset root.
ASTRALX_A10K_SIMPHY_SUBDIR="10k-simphy"

# Name of the merged CSV written by collect-scores-a10k.sh.
ASTRALX_A10K_MERGED_CSV_NAME="a10k_astralx_scores_merged.csv"

# Small dataset-level provenance files copied to the mirror root when present:
# <data>/<dataset><suffix>. None of them are produced by this repository, so
# their absence is not an error.
ASTRALX_A10K_RECORD_SUFFIXES=(".command" ".source" ".params")

# Replicate-level record written by run-a10k.sh when it roots estimated gene
# trees; copied beside the replicate's mirrored results.
ASTRALX_A10K_ROOTING_RECORD_NAME="estimatedgenetrees.rooted.command"

# Files that must never appear in the outputs mirror: the (rooted) estimated
# gene trees, the true gene trees, the species trees, and archives.
ASTRALX_A10K_FORBIDDEN_MIRROR_NAMES=(
  "estimatedgenetrees*.tre" "truegenetrees" "truegenetrees*.tre" "truegenetrees*.trees"
  "s_tree.trees" "*.trees" "all_gt.tre"
  "*.db" "*.db-journal" "*.zip" "*.tar" "*.tar.gz" "*.tgz"
)

# Print the default outputs mirror root for a resolved A10K dataset directory.
astralx_default_a10k_outputs_dir() {
  local data_dir="${1%/}"
  if [[ -z "$data_dir" ]]; then
    astralx__outputs_error "an A10K data directory is required to derive the outputs directory."
    return 2
  fi
  # <parent>/<dataset> -> <parent>/outputs/<dataset>
  printf '%s/outputs/%s\n' "$(dirname -- "$data_dir")" "$(basename -- "$data_dir")"
}

# Resolve, validate, and create the outputs mirror root.
#   astralx_prepare_a10k_outputs_dir REQUESTED_OUTPUTS_DIR RESOLVED_DATA_DIR
# An empty REQUESTED_OUTPUTS_DIR selects the default described above.
astralx_prepare_a10k_outputs_dir() {
  local requested_dir="${1:-}" data_dir="${2:-}"
  local resolved_dir

  if [[ -z "$data_dir" ]]; then
    astralx__outputs_error "internal: resolved A10K data directory is required."
    return 2
  fi
  if [[ -z "$requested_dir" ]]; then
    requested_dir="$(astralx_default_a10k_outputs_dir "$data_dir")" || return 2
  fi
  if [[ "$requested_dir" == "~/"* ]]; then
    requested_dir="${HOME}/${requested_dir:2}"
  fi
  if [[ -e "$requested_dir" && ! -d "$requested_dir" ]]; then
    astralx__outputs_error "A10K outputs path exists but is not a directory: $requested_dir"
    return 2
  fi

  resolved_dir="$(realpath -m -- "$requested_dir")" || {
    astralx__outputs_error "could not resolve A10K outputs directory: $requested_dir"
    return 2
  }
  if astralx__path_is_within "$resolved_dir" "$data_dir"; then
    astralx__outputs_error "A10K outputs directory must not be the data directory or inside it: $resolved_dir"
    return 2
  fi
  if astralx__path_is_within "$data_dir" "$resolved_dir"; then
    astralx__outputs_error "A10K outputs directory must not contain the data directory: $resolved_dir"
    return 2
  fi
  if ! mkdir -p -- "$resolved_dir"; then
    astralx__outputs_error "could not create A10K outputs directory: $resolved_dir"
    return 2
  fi
  if ! resolved_dir="$(cd "$resolved_dir" && pwd -P)"; then
    astralx__outputs_error "could not enter A10K outputs directory: $requested_dir"
    return 2
  fi
  printf '%s\n' "$resolved_dir"
}

# Print every forbidden file found beneath a directory (one per line).
astralx_a10k_find_forbidden_in_mirror() {
  local root="$1"
  local -a find_args=()
  local pattern first=true
  for pattern in "${ASTRALX_A10K_FORBIDDEN_MIRROR_NAMES[@]}"; do
    if [[ "$first" == true ]]; then
      first=false
    else
      find_args+=(-o)
    fi
    find_args+=(-name "$pattern")
  done
  find "$root" -type f \( "${find_args[@]}" \) -print 2>/dev/null
}

# Copy one small file into the mirror unless an identical copy is there.
astralx__a10k_copy_if_changed() {
  local source="$1" destination="$2"
  [[ -f "$source" ]] || return 1
  if [[ -f "$destination" ]] && cmp -s -- "$source" "$destination"; then
    return 0
  fi
  mkdir -p -- "$(dirname -- "$destination")" || return 1
  astralx__copy_small_file_atomic "$source" "$destination"
}

# Copy dataset-level provenance files (<data>/<dataset>.command etc. and a
# README) to the mirror root. Returns 0 always; these files are optional.
#   astralx_mirror_a10k_dataset_records DATA_DIR OUTPUTS_ROOT
astralx_mirror_a10k_dataset_records() {
  local data_dir="${1%/}" outputs_root="${2%/}"
  local dataset suffix source readme
  dataset="$(basename -- "$data_dir")"
  for suffix in "${ASTRALX_A10K_RECORD_SUFFIXES[@]}"; do
    source="${data_dir}/${dataset}${suffix}"
    [[ -f "$source" ]] || continue
    astralx__a10k_copy_if_changed "$source" "${outputs_root}/${dataset}${suffix}" ||
      astralx__outputs_error "could not copy $source into the A10K mirror"
  done
  for readme in "$data_dir"/README "$data_dir"/README.*; do
    [[ -f "$readme" ]] || continue
    astralx__a10k_copy_if_changed "$readme" "${outputs_root}/$(basename -- "$readme")" ||
      astralx__outputs_error "could not copy $readme into the A10K mirror"
  done
  return 0
}

# Copy the merged scores CSV (written by collect-scores-a10k.sh) to the mirror
# root. Returns 1 when the CSV does not exist.
#   astralx_mirror_a10k_merged_csv DATA_DIR OUTPUTS_ROOT
astralx_mirror_a10k_merged_csv() {
  local data_dir="${1%/}" outputs_root="${2%/}"
  local source="${data_dir}/${ASTRALX_A10K_MERGED_CSV_NAME}"
  [[ -f "$source" ]] || return 1
  astralx__a10k_copy_if_changed "$source" "${outputs_root}/${ASTRALX_A10K_MERGED_CSV_NAME}"
}

# Split a results directory into its mirror components.
#   astralx_a10k_results_components DATA_DIR RESULTS_DIR
# Prints five lines: dataset subdir (10k-simphy), replicate, method-outputs dir
# name, tree type, setting. Fails when RESULTS_DIR does not have the shape
# <data>/10k-simphy/<replicate>/<method>_outputs/<tree_type>/<setting>.
astralx_a10k_results_components() {
  local data_dir="${1%/}" results_dir="${2%/}"
  local relative subdir replicate method_dir tree_type setting rest part

  results_dir="$(realpath -m -- "$results_dir")"
  if ! astralx__path_is_within "$results_dir" "$data_dir" || [[ "$results_dir" == "$data_dir" ]]; then
    astralx__outputs_error "results directory is not inside the A10K data directory: $results_dir"
    return 1
  fi
  relative="${results_dir#"$data_dir"/}"

  IFS='/' read -r subdir replicate method_dir tree_type setting rest <<< "$relative"
  if [[ -z "$subdir" || -z "$replicate" || -z "$method_dir" || -z "$tree_type" || -z "$setting" || -n "${rest:-}" ]]; then
    astralx__outputs_error "expected ${ASTRALX_A10K_SIMPHY_SUBDIR}/<replicate>/<method>_outputs/<tree_type>/<setting>, got: $relative"
    return 1
  fi
  if [[ "$subdir" != "$ASTRALX_A10K_SIMPHY_SUBDIR" ]]; then
    astralx__outputs_error "expected results beneath '${ASTRALX_A10K_SIMPHY_SUBDIR}', got '$subdir' in: $relative"
    return 1
  fi
  if [[ "$method_dir" != *_outputs ]]; then
    astralx__outputs_error "expected a '<method>_outputs' directory, got '$method_dir' in: $relative"
    return 1
  fi
  for part in "$subdir" "$replicate" "$method_dir" "$tree_type" "$setting"; do
    if [[ "$part" == . || "$part" == .. ]]; then
      astralx__outputs_error "unsafe path component in: $relative"
      return 1
    fi
  done

  printf '%s\n%s\n%s\n%s\n%s\n' "$subdir" "$replicate" "$method_dir" "$tree_type" "$setting"
}

# Mirror one results directory into the outputs tree.
#   astralx_mirror_a10k_results DATA_DIR OUTPUTS_ROOT RESULTS_DIR
# The mirror leaf is replaced atomically so it always equals the source leaf.
# The replicate's rooting record and the dataset records are refreshed
# alongside. Prints the mirror leaf path.
astralx_mirror_a10k_results() {
  local data_dir="${1%/}" outputs_root="${2%/}" results_dir="${3%/}"
  local -a parts=()
  local subdir replicate method_dir tree_type setting
  local mirror_replicate_dir mirror_leaf forbidden rooting_record

  if [[ ! -d "$results_dir" ]]; then
    astralx__outputs_error "results directory does not exist: $results_dir"
    return 1
  fi
  mapfile -t parts < <(astralx_a10k_results_components "$data_dir" "$results_dir") || return 1
  [[ ${#parts[@]} -eq 5 ]] || return 1
  subdir="${parts[0]}"; replicate="${parts[1]}"; method_dir="${parts[2]}"
  tree_type="${parts[3]}"; setting="${parts[4]}"

  forbidden="$(astralx_a10k_find_forbidden_in_mirror "$results_dir" | head -n1)"
  if [[ -n "$forbidden" ]]; then
    astralx__outputs_error "refusing to mirror a results directory containing A10K input data: $forbidden"
    return 1
  fi

  mirror_replicate_dir="${outputs_root}/${method_dir}/${subdir}/${replicate}"
  mirror_leaf="${mirror_replicate_dir}/${tree_type}/${setting}"
  mkdir -p -- "${mirror_replicate_dir}/${tree_type}" || return 1

  astralx_mirror_a10k_dataset_records "$data_dir" "$outputs_root"
  rooting_record="${data_dir}/${subdir}/${replicate}/estimatedgenetrees/${ASTRALX_A10K_ROOTING_RECORD_NAME}"
  if [[ -f "$rooting_record" ]]; then
    astralx__a10k_copy_if_changed "$rooting_record" "${mirror_replicate_dir}/${ASTRALX_A10K_ROOTING_RECORD_NAME}" ||
      astralx__outputs_error "could not copy $rooting_record into the A10K mirror"
  fi

  astralx_mirror_directory_atomic "$results_dir" "$mirror_leaf" || return 1
  printf '%s\n' "$mirror_leaf"
}

# Print every results directory beneath an A10K data directory, NUL-separated:
#   <data>/10k-simphy/<replicate>/<method>_outputs/<tree_type>/<setting>
astralx_list_a10k_results_dirs() {
  local data_dir="${1%/}"
  find "${data_dir}/${ASTRALX_A10K_SIMPHY_SUBDIR}" -mindepth 4 -maxdepth 4 -type d \
    -path "*/*_outputs/*/*" -not -name '.*' -print0 2>/dev/null | sort -zV
}

# Resolve the outputs mirror root without a data directory (for tools that only
# read the mirror). An explicit path is required here because the A10K dataset
# location is always given explicitly.
astralx_prepare_a10k_outputs_dir_standalone() {
  local requested_dir="${1:-}" resolved_dir

  if [[ -z "$requested_dir" ]]; then
    astralx__outputs_error "an A10K outputs directory is required (pass --outputs-dir, or --data-dir so it can be derived)."
    return 2
  fi
  if [[ "$requested_dir" == "~/"* ]]; then
    requested_dir="${HOME}/${requested_dir:2}"
  fi
  if [[ -e "$requested_dir" && ! -d "$requested_dir" ]]; then
    astralx__outputs_error "A10K outputs path exists but is not a directory: $requested_dir"
    return 2
  fi
  if ! mkdir -p -- "$requested_dir"; then
    astralx__outputs_error "could not create A10K outputs directory: $requested_dir"
    return 2
  fi
  if ! resolved_dir="$(cd "$requested_dir" && pwd -P)"; then
    astralx__outputs_error "could not resolve A10K outputs directory: $requested_dir"
    return 2
  fi
  printf '%s\n' "$resolved_dir"
}
