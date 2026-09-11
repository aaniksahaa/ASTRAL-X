#!/usr/bin/env bash

# Reproducibility mirror for simulated-run outputs.
#
# Every inference run on a SimPhy replicate keeps writing its results inside the
# data tree exactly as before:
#
#   <data>/<dataset>/<replicate>/<method>_outputs/<setting>/...
#
# The functions below additionally maintain a compact, shareable copy that
# contains only the small artifacts (inferred trees, CSVs, run markers/logs) plus
# the SimPhy command files, and never the gene trees or true species trees:
#
#   <outputs>/<method>_outputs/<dataset>/<dataset>.command
#   <outputs>/<method>_outputs/<dataset>/<dataset>.params
#   <outputs>/<method>_outputs/<dataset>/<replicate>/<setting>/...
#
# <outputs> defaults to $PHYLOGENY_DATA_DIR/outputs/simphy for the standard
# $PHYLOGENY_DATA_DIR/simphy/data tree (".../simphy/data" -> ".../outputs/simphy").
# Any other data directory named "data" mirrors into its "outputs" sibling, and
# any other explicit data directory into "<data-dir>_outputs", unless an outputs
# directory is given explicitly.

# Files that describe how a dataset was simulated. Copied per dataset.
ASTRALX_SIMPHY_COMMAND_SUFFIXES=(".command" ".params")

# Files that must never appear in the outputs mirror. These are the large
# simulated inputs and SimPhy databases that stay in the data tree only.
ASTRALX_SIMPHY_FORBIDDEN_MIRROR_NAMES=(
  "all_gt.tre" "s_tree.trees" "l_trees.trees" "g_trees*.trees"
  "*.db" "*.db-journal" "*.zip" "stat-sim.csv"
)

astralx__outputs_error() {
  echo "Error: $*" >&2
}

astralx__path_is_within() {
  # astralx__path_is_within CHILD PARENT -> true when CHILD == PARENT or is below it.
  local child="${1%/}" parent="${2%/}"
  [[ "$child" == "$parent" || "$child" == "$parent"/* ]]
}

# Print the default outputs mirror root for a resolved SimPhy data directory.
astralx_default_simphy_outputs_dir() {
  local data_dir="${1%/}"
  if [[ -z "$data_dir" ]]; then
    astralx__outputs_error "a SimPhy data directory is required to derive the outputs directory."
    return 2
  fi
  local parent_dir
  parent_dir="$(dirname -- "$data_dir")"
  if [[ "$(basename -- "$data_dir")" == "data" && "$(basename -- "$parent_dir")" == "simphy" ]]; then
    # Standard layout: <root>/simphy/data -> <root>/outputs/simphy
    printf '%s/outputs/simphy\n' "$(dirname -- "$parent_dir")"
  elif [[ "$(basename -- "$data_dir")" == "data" ]]; then
    printf '%s/outputs\n' "$parent_dir"
  else
    printf '%s_outputs\n' "$data_dir"
  fi
}

# Resolve, validate, and create the outputs mirror root.
#   astralx_prepare_simphy_outputs_dir REQUESTED_OUTPUTS_DIR RESOLVED_DATA_DIR
# An empty REQUESTED_OUTPUTS_DIR selects the default described above.
astralx_prepare_simphy_outputs_dir() {
  local requested_dir="${1:-}" data_dir="${2:-}"
  local resolved_dir

  if [[ -z "$data_dir" ]]; then
    astralx__outputs_error "internal: resolved SimPhy data directory is required."
    return 2
  fi
  if [[ -z "$requested_dir" ]]; then
    requested_dir="$(astralx_default_simphy_outputs_dir "$data_dir")" || return 2
  fi
  if [[ "$requested_dir" == "~/"* ]]; then
    requested_dir="${HOME}/${requested_dir:2}"
  fi
  if [[ -e "$requested_dir" && ! -d "$requested_dir" ]]; then
    astralx__outputs_error "SimPhy outputs path exists but is not a directory: $requested_dir"
    return 2
  fi

  resolved_dir="$(realpath -m -- "$requested_dir")" || {
    astralx__outputs_error "could not resolve SimPhy outputs directory: $requested_dir"
    return 2
  }
  if astralx__path_is_within "$resolved_dir" "$data_dir"; then
    astralx__outputs_error "SimPhy outputs directory must not be the data directory or inside it: $resolved_dir"
    return 2
  fi
  if astralx__path_is_within "$data_dir" "$resolved_dir"; then
    astralx__outputs_error "SimPhy outputs directory must not contain the data directory: $resolved_dir"
    return 2
  fi
  if ! mkdir -p -- "$resolved_dir"; then
    astralx__outputs_error "could not create SimPhy outputs directory: $resolved_dir"
    return 2
  fi
  if ! resolved_dir="$(cd "$resolved_dir" && pwd -P)"; then
    astralx__outputs_error "could not enter SimPhy outputs directory: $requested_dir"
    return 2
  fi
  printf '%s\n' "$resolved_dir"
}

# True when a file name matches one of the forbidden mirror patterns.
astralx_simphy_name_is_forbidden_in_mirror() {
  local name="$1" pattern
  for pattern in "${ASTRALX_SIMPHY_FORBIDDEN_MIRROR_NAMES[@]}"; do
    # shellcheck disable=SC2053
    [[ "$name" == $pattern ]] && return 0
  done
  return 1
}

# Print every forbidden file found beneath a directory (one per line).
astralx_simphy_find_forbidden_in_mirror() {
  local root="$1"
  local -a find_args=()
  local pattern first=true
  for pattern in "${ASTRALX_SIMPHY_FORBIDDEN_MIRROR_NAMES[@]}"; do
    if [[ "$first" == true ]]; then
      first=false
    else
      find_args+=(-o)
    fi
    find_args+=(-name "$pattern")
  done
  find "$root" -type f \( "${find_args[@]}" \) -print 2>/dev/null
}

# Copy one small file atomically (temp file + rename), preserving timestamps.
astralx__copy_small_file_atomic() {
  local source="$1" destination="$2" temp
  temp="$(mktemp -- "${destination}.tmp.XXXXXX")" || return 1
  if cp -p -- "$source" "$temp" && mv -f -- "$temp" "$destination"; then
    return 0
  fi
  rm -f -- "$temp"
  return 1
}

# Copy the SimPhy command/params files of a dataset into its mirror directory.
#   astralx_mirror_simphy_dataset_commands DATA_DIR MIRROR_DATASET_DIR DATASET_NAME
# For "<name>_incomplete" datasets the base dataset's files are copied as well,
# because the incomplete variant is derived from that simulation.
# Returns 0 when at least one .command file was mirrored, 1 otherwise.
astralx_mirror_simphy_dataset_commands() {
  local data_dir="${1%/}" mirror_dataset_dir="${2%/}" dataset_name="$3"
  local -a names=("$dataset_name")
  local name suffix source destination copied_command=false

  if [[ "$dataset_name" == *_incomplete ]]; then
    names+=("${dataset_name%_incomplete}")
  fi

  mkdir -p -- "$mirror_dataset_dir" || return 1
  for name in "${names[@]}"; do
    for suffix in "${ASTRALX_SIMPHY_COMMAND_SUFFIXES[@]}"; do
      source="${data_dir}/${name}/${name}${suffix}"
      destination="${mirror_dataset_dir}/${name}${suffix}"
      [[ -f "$source" ]] || continue
      if [[ -f "$destination" ]] && cmp -s -- "$source" "$destination"; then
        [[ "$suffix" == ".command" ]] && copied_command=true
        continue
      fi
      if astralx__copy_small_file_atomic "$source" "$destination"; then
        [[ "$suffix" == ".command" ]] && copied_command=true
      else
        astralx__outputs_error "could not copy $source to $destination"
      fi
    done
  done

  [[ "$copied_command" == true ]]
}

# Split a results directory into its mirror components.
#   astralx_simphy_results_components DATA_DIR RESULTS_DIR
# Prints four lines: dataset, replicate, method-outputs dir name, setting.
# Fails when RESULTS_DIR does not have the expected shape beneath DATA_DIR.
astralx_simphy_results_components() {
  local data_dir="${1%/}" results_dir="${2%/}"
  local relative dataset replicate method_dir setting rest

  results_dir="$(realpath -m -- "$results_dir")"
  if ! astralx__path_is_within "$results_dir" "$data_dir" || [[ "$results_dir" == "$data_dir" ]]; then
    astralx__outputs_error "results directory is not inside the SimPhy data directory: $results_dir"
    return 1
  fi
  relative="${results_dir#"$data_dir"/}"

  dataset="${relative%%/*}"; rest="${relative#*/}"
  [[ "$rest" != "$relative" ]] || { astralx__outputs_error "unexpected results layout: $relative"; return 1; }
  replicate="${rest%%/*}"; rest="${rest#*/}"
  [[ "$rest" != "$replicate" ]] || { astralx__outputs_error "unexpected results layout: $relative"; return 1; }
  method_dir="${rest%%/*}"; setting="${rest#*/}"
  if [[ "$setting" == "$method_dir" || -z "$setting" || "$setting" == */* ]]; then
    astralx__outputs_error "expected <dataset>/<replicate>/<method>_outputs/<setting>, got: $relative"
    return 1
  fi
  if [[ "$method_dir" != *_outputs ]]; then
    astralx__outputs_error "expected a '<method>_outputs' directory, got '$method_dir' in: $relative"
    return 1
  fi
  for rest in "$dataset" "$replicate" "$method_dir" "$setting"; do
    if [[ "$rest" == . || "$rest" == .. || -z "$rest" ]]; then
      astralx__outputs_error "unsafe path component in: $relative"
      return 1
    fi
  done

  printf '%s\n%s\n%s\n%s\n' "$dataset" "$replicate" "$method_dir" "$setting"
}

# Mirror one results directory into the outputs tree.
#   astralx_mirror_simulated_results DATA_DIR OUTPUTS_ROOT RESULTS_DIR
# The mirror leaf is replaced atomically so it always equals the source leaf.
# Dataset command files are refreshed alongside. Prints the mirror leaf path.
astralx_mirror_simulated_results() {
  local data_dir="${1%/}" outputs_root="${2%/}" results_dir="${3%/}"
  local -a parts=()
  local dataset replicate method_dir setting
  local mirror_dataset_dir mirror_leaf mirror_parent temp_new temp_old forbidden

  if [[ ! -d "$results_dir" ]]; then
    astralx__outputs_error "results directory does not exist: $results_dir"
    return 1
  fi
  mapfile -t parts < <(astralx_simphy_results_components "$data_dir" "$results_dir") || return 1
  [[ ${#parts[@]} -eq 4 ]] || return 1
  dataset="${parts[0]}"; replicate="${parts[1]}"; method_dir="${parts[2]}"; setting="${parts[3]}"

  forbidden="$(astralx_simphy_find_forbidden_in_mirror "$results_dir" | head -n1)"
  if [[ -n "$forbidden" ]]; then
    astralx__outputs_error "refusing to mirror a results directory containing simulated input data: $forbidden"
    return 1
  fi

  mirror_dataset_dir="${outputs_root}/${method_dir}/${dataset}"
  mirror_parent="${mirror_dataset_dir}/${replicate}"
  mirror_leaf="${mirror_parent}/${setting}"

  mkdir -p -- "$mirror_parent" || return 1
  if ! astralx_mirror_simphy_dataset_commands "$data_dir" "$mirror_dataset_dir" "$dataset"; then
    echo "Warning: no SimPhy .command file found for dataset '$dataset'; the mirror lacks its simulation command." >&2
  fi

  temp_new="$(mktemp -d -- "${mirror_parent}/.${setting}.mirror.XXXXXX")" || return 1
  if ! cp -a -- "${results_dir}/." "${temp_new}/"; then
    rm -rf -- "$temp_new"
    astralx__outputs_error "could not copy results into the outputs mirror: $results_dir"
    return 1
  fi

  temp_old=""
  if [[ -e "$mirror_leaf" ]]; then
    temp_old="$(mktemp -d -u -- "${mirror_parent}/.${setting}.previous.XXXXXX")"
    if ! mv -- "$mirror_leaf" "$temp_old"; then
      rm -rf -- "$temp_new"
      astralx__outputs_error "could not replace the existing mirror: $mirror_leaf"
      return 1
    fi
  fi
  if ! mv -- "$temp_new" "$mirror_leaf"; then
    [[ -n "$temp_old" && -d "$temp_old" ]] && mv -- "$temp_old" "$mirror_leaf" 2>/dev/null
    rm -rf -- "$temp_new"
    astralx__outputs_error "could not install the outputs mirror: $mirror_leaf"
    return 1
  fi
  [[ -n "$temp_old" ]] && rm -rf -- "$temp_old"

  printf '%s\n' "$mirror_leaf"
}

# Print every results directory beneath a data directory, NUL-separated:
#   <data>/<dataset>/<replicate>/<method>_outputs/<setting>
astralx_list_simulated_results_dirs() {
  local data_dir="${1%/}"
  find "$data_dir" -mindepth 4 -maxdepth 4 -type d -path '*/*_outputs/*' \
    -not -name '.*' -print0 2>/dev/null | sort -z
}

# Resolve the outputs mirror root without a data directory (for tools that only
# read the mirror). An explicit path wins; otherwise PHYLOGENY_DATA_DIR is
# required and <PHYLOGENY_DATA_DIR>/outputs/simphy is used and created.
astralx_prepare_simphy_outputs_dir_standalone() {
  local requested_dir="${1:-}" resolved_dir

  if [[ -z "$requested_dir" ]]; then
    if [[ -z "${PHYLOGENY_DATA_DIR:-}" ]]; then
      astralx__outputs_error "PHYLOGENY_DATA_DIR is not set. Set it or pass an explicit SimPhy outputs directory."
      return 2
    fi
    requested_dir="${PHYLOGENY_DATA_DIR%/}/outputs/simphy"
  fi
  if [[ "$requested_dir" == "~/"* ]]; then
    requested_dir="${HOME}/${requested_dir:2}"
  fi
  if [[ -e "$requested_dir" && ! -d "$requested_dir" ]]; then
    astralx__outputs_error "SimPhy outputs path exists but is not a directory: $requested_dir"
    return 2
  fi
  if ! mkdir -p -- "$requested_dir"; then
    astralx__outputs_error "could not create SimPhy outputs directory: $requested_dir"
    return 2
  fi
  if ! resolved_dir="$(cd "$requested_dir" && pwd -P)"; then
    astralx__outputs_error "could not resolve SimPhy outputs directory: $requested_dir"
    return 2
  fi
  printf '%s\n' "$resolved_dir"
}

# Validate a simulated dataset directory name as it appears in the outputs tree.
# Sets BASH_REMATCH: [1]=taxa [2]=gene trees [3]=sb [6]=spmin [7]=spmax
# [8]="_incomplete" or empty.
astralx_simphy_dataset_name_is_valid() {
  local name="$1"
  [[ "$name" =~ ^t_([1-9][0-9]*)_g_([1-9][0-9]*)_sb_([0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?)_spmin_([1-9][0-9]*)_spmax_([1-9][0-9]*)(_incomplete)?$ ]]
}
