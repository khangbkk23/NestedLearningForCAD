# External baseline reference

ReplayCAD runs from an explicit `REPLAYCAD_ROOT`, with separate detector/LDM environments and assets. Its upstream is https://github.com/HULEI7/ReplayCAD (also linked by the IJCAI paper). Khang gitlink6d09dc3 differs from the observed official HEAD acc6195; this repository does not silently substitute either commit. `replaycad.json` preserves both identities. Source/assets and full continual-boundary reproduction are not supplied or claimed by this scaffold.

The old tree had a gitlink without `.gitmodules`; the current tree uses this explicit external reference instead of a broken empty submodule. Existing native orchestration and tests remain.
