# Third-party sources and notices

Jevduck builds on the Microduck simulator and includes upstream robot assets. This file records their provenance and the license information published by their authors. It does not assign a new license to those materials or extend one upstream project's terms to another project.

## Microduck browser simulator

The files under `vendor/microduck-simulator/` originate from the [Pollen Robotics Microduck simulator on Hugging Face](https://huggingface.co/spaces/pollen-robotics/microduck-simulator/tree/023172c8a7d629b5258d90364c13bafe013abbfa), pinned at revision:

```text
023172c8a7d629b5258d90364c13bafe013abbfa
```

Jevduck modifies this simulator to support measured task execution and shared physics for multiple robots. The [vendored README](vendor/microduck-simulator/README.md) describes the upstream project; the [root README](README.md) describes this application.

At this revision, the Space's repository metadata and README do not declare a license. Its repository file listing also lacks a standalone license or notice file. This can be inspected through the [pinned Space API response](https://huggingface.co/api/spaces/pollen-robotics/microduck-simulator/revision/023172c8a7d629b5258d90364c13bafe013abbfa). The Apache license of a related robot repository must not be assumed to cover all Space code or assets. No blanket license for those files is asserted here.

The [asset provenance manifest](vendor/microduck-simulator/ASSET_PROVENANCE.json) records the source revision and SHA-256 hashes for the included upstream assets. These hashes establish file identity; they do not establish licensing permission. See [validation and reproduction](docs/VALIDATION.md) for the verification procedure.

## Related Microduck projects

The upstream simulator credits [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) and [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) for its policies and MJCF model.

- **Microduck runtime:** its [published license](https://github.com/pollen-robotics/microduck/blob/main/LICENSE) is Apache License 2.0.
- **Microduck RL:** its [published license](https://github.com/pollen-robotics/microduck_rl/blob/develop/LICENSE) is Apache License 2.0. Its [README license section](https://github.com/pollen-robotics/microduck_rl#license) separately states: “3D model files are licensed under Creative Commons BY-SA-NC.” That statement does not specify a Creative Commons version.

These are source-project declarations, checked on 2026-09-18. They provide context for the vendored materials, not a file-by-file determination of the terms governing every mesh or exported policy. The pinned Space manifest identifies the exact copies included here. Upstream materials remain subject to their respective authors' terms, including any separate asset terms.

## ZzFX

[`vendor/microduck-simulator/app/src/game/vendor/zzfx.js`](vendor/microduck-simulator/app/src/game/vendor/zzfx.js) retains attribution to **ZzFX v1.3.2 by Frank Force**, copyright 2019 Frank Force, and its MIT license notice. Its header also describes the upstream simulator's adaptation of synthesis and playback. The original project is [KilledByAPixel/ZzFX](https://github.com/KilledByAPixel/ZzFX).

## Package dependencies

JavaScript packages are installed from the root and simulator `package-lock.json` files. Their own licenses and notices apply. Generated dependency directories and compiled bundles are not source assets of this repository.

This repository does not currently declare a blanket license for Jevduck's original application code. The absence of such a declaration does not replace or remove the notices attached to upstream material.
