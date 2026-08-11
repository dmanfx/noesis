# `nvdsroiexclude` (DS8)

`nvdsroiexclude` is the canonical pre-tracker metadata pruning element for
static exclusion polygons. It validates and hashes the complete owner-managed
INI before activation, rejects unsafe or ambiguous input, and retains the last
validated in-memory configuration when a live reload is rejected.

The DS8 and DS9 C++ sources are intentionally byte-identical owned mirrors.
Their CMake projects and binaries are separate: this directory only accepts
DeepStream major 8 headers and libraries, while `DS9/csrc/nvdsroiexclude/`
only accepts DeepStream major 9.

Build and inspect the DS8-owned binary without installing it globally:

```bash
./gst-plugins/build_nvdsroiexclude.sh
GST_PLUGIN_PATH="$PWD/gst-plugins" gst-inspect-1.0 nvdsroiexclude
```

Do not copy this binary into DS9, and do not use the DS9 binary in DS8.
