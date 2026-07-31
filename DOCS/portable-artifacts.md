# Portable ASTRAL-X artifacts

`build_portable.sh` and `build_portable.ps1` create self-contained application
images. The target machine does not need Java, a JDK, or a CUDA toolkit. Extract
the archive and invoke `astralx` (`astralx.exe` on Windows).

The application image is deliberately an archive containing a native launcher,
a trimmed HotSpot runtime, the ASTRAL-X classes, and optional CUDA libraries. A
literal one-file executable would have to unpack the JVM and native libraries at
runtime, which is less robust on clusters, read-only filesystems, and machines
with restrictive security policies.

## Artifact set

| Canonical artifact | CPU | CUDA | Release validation runner |
|---|---:|---:|---|
| Linux x86-64 | yes | yes, optional at runtime | native NVIDIA runner |
| Linux ARM64 | yes | yes, optional at runtime | native ARM64 NVIDIA runner |
| Windows x86-64 | yes | yes, optional at runtime | native Windows NVIDIA runner |
| macOS Intel | yes | unavailable | native GitHub runner |
| macOS Apple Silicon | yes | unavailable | native GitHub runner |

macOS has no current NVIDIA CUDA runtime, so its artifacts are CPU-only. A
machine containing an FPGA can run the CPU artifact; ASTRAL-X does not currently
contain an FPGA backend.

One executable cannot span OS executable formats or CPU instruction sets. The
release therefore publishes exactly one checksummed artifact per OS/CPU family—not
separate CPU and CUDA editions. The Linux and Windows images bundle CUDA but retain
the complete CPU path; CUDA is selected only when its startup probe succeeds. This is
not Java's old "install a JVM first" deployment model: each artifact carries its
own matching runtime.

## Build locally

Linux release artifact (CUDA bundled, automatic CPU fallback):

```bash
./build_portable.sh
```

macOS release artifact (CPU, because CUDA is unavailable):

```bash
./build_portable.sh --without-cuda
```

Windows PowerShell release artifact:

```powershell
.\build_portable.ps1
```

`--without-cuda` (`-WithoutCuda` on Windows) creates a CPU-only build. The
existing `--cpu-only`/`-CpuOnly` spelling is equivalent. A CPU-only build is not
a separate public edition: it uses the same canonical artifact name, so a
same-version rebuild requires the explicit force option. Release CI uses
`--with-cuda` as a strict check so a missing CUDA toolkit fails the release
instead of silently producing a CPU-only archive.

Building requires JDK 21 or newer. CUDA builds additionally require `nvcc` and a
supported host C/C++ compiler. These are build-machine requirements only.
Artifacts, SHA-256 files, and concise JSON manifests are written under
`dist/<version>/`. Pass `--version 1.2.0` (`-Version 1.2.0` on Windows) to build
an explicit release version. Existing versions remain side-by-side; rebuilding
the same version/platform is refused unless `--force` (`-Force`) is supplied.
The local `dist/` directory is ignored by Git; release automation publishes the
archives, checksums, and manifests as GitHub Release assets.

The default CUDA architecture, `all-major`, embeds native code for every major
GPU generation supported by the installed toolkit and PTX for forward
compatibility. The generated library also records its minimum compute
capability. If a GPU is older, its driver is missing/incompatible, or CUDA cannot
be loaded, the default `--auto` mode explains the reason and selects the exact
CPU implementation before computation starts. `--gpu-strict` converts that
fallback into an immediate, diagnostic failure for environments that require
CUDA.

## Run and diagnose

```bash
./astralx --help
./astralx --diagnose
./astralx -i gene_trees.tre -o species_tree.tre
```

Every artifact includes a ready-made 37-taxon example:

```bash
./astralx -i example/all_gt_37.tre \
  -o example/predicted_st_37.tre \
  --search-space S1 -vv
```

`example/all_gt_37.tre` contains the gene trees and `example/true_37.tre` is the
reference species tree. Running the command creates
`example/predicted_st_37.tre`, keeping the input, truth, and inferred tree
together in one directory.

`--diagnose` does not require an input tree. It reports the OS/architecture,
bundled runtime, heap allowance, native-library status, CUDA driver/runtime,
device compute capability, and the selected CPU/GPU mode. Fatal Java errors also
produce an `astralx-crash-*.log` containing the current phase and runtime details.
An OS-level kill (for example, the Linux OOM killer) or a hardware reset can end
a process before any program is able to write a report; system logs remain the
source of truth in that case.

## Release automation and reliability

`.github/workflows/portable-artifacts.yml` builds the five canonical artifacts
on native runners, executes packaged CPU smoke tests everywhere, and runs a real
CUDA inference on the Linux and Windows NVIDIA runners. It publishes the complete
one-per-platform set for version tags. This intentionally avoids calling an
unexecuted CUDA cross-compile "tested."

For a broad Linux compatibility baseline, use the oldest supported release
runner/container: glibc binaries are backward-incompatible when built on a newer
distro. The artifact manifest and `BUILD-INFO.txt` record the highest required
glibc symbol version, along with the build platform/toolchains. Windows and
macOS may display SmartScreen/Gatekeeper warnings until release signing
certificates are configured; checksums verify integrity but are not a substitute
for code signing.
