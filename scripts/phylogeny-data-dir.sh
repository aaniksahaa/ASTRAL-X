#!/usr/bin/env bash

# Resolve the shared SimPhy data root. An explicit path takes precedence;
# otherwise PHYLOGENY_DATA_DIR is required and the standard subdirectory is
# created beneath it.
astralx_prepare_simphy_data_dir() {
  local requested_dir="${1:-}"
  local resolved_dir

  if [[ -z "$requested_dir" ]]; then
    if [[ -z "${PHYLOGENY_DATA_DIR:-}" ]]; then
      echo "Error: PHYLOGENY_DATA_DIR is not set. Set it or pass an explicit SimPhy data directory." >&2
      return 2
    fi
    requested_dir="${PHYLOGENY_DATA_DIR%/}/simphy/data"
  fi

  if [[ "$requested_dir" == "~/"* ]]; then
    requested_dir="${HOME}/${requested_dir:2}"
  fi
  if [[ -e "$requested_dir" && ! -d "$requested_dir" ]]; then
    echo "Error: SimPhy data path exists but is not a directory: $requested_dir" >&2
    return 2
  fi
  if ! mkdir -p -- "$requested_dir"; then
    echo "Error: could not create SimPhy data directory: $requested_dir" >&2
    return 2
  fi
  if ! resolved_dir="$(cd "$requested_dir" && pwd -P)"; then
    echo "Error: could not resolve SimPhy data directory: $requested_dir" >&2
    return 2
  fi

  printf '%s\n' "$resolved_dir"
}

# Same as above, but with a repository-local fallback for checkouts that do not
# set PHYLOGENY_DATA_DIR. Every script that reads or writes the SimPhy data tree
# resolves it through this function so they all agree on one location:
#   explicit path  >  $PHYLOGENY_DATA_DIR/simphy/data  >  FALLBACK
astralx_resolve_simphy_data_dir() {
  local requested_dir="${1:-}" fallback_dir="${2:-}"

  if [[ -z "$requested_dir" && -z "${PHYLOGENY_DATA_DIR:-}" ]]; then
    requested_dir="$fallback_dir"
  fi
  astralx_prepare_simphy_data_dir "$requested_dir"
}
